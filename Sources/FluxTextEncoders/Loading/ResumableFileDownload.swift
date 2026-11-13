// ResumableFileDownload.swift - Stream one HTTP body into a resumable `.part` file
//
// Shared by Flux2ModelDownloader (Flux2Core) and TextEncoderModelDownloader (this module) so both
// the diffusion components and the Qwen3 text encoder keep every received byte across a cancel,
// pause or network drop.

import Foundation

package enum ResumableFileDownload {
    package enum Failure: Error, LocalizedError {
        case notHTTP(String)
        case badStatus(String, Int)
        case rangeMismatch(String, expectedStart: Int64, header: String?)

        package var errorDescription: String? {
            switch self {
            case .notHTTP(let path):
                return "Failed to download \(path)"
            case .badStatus(let path, let code):
                return "Failed to download \(path): HTTP \(code)"
            case .rangeMismatch(let path, let start, let header):
                return "Failed to download \(path): partial response \(header ?? "without Content-Range") does not continue at byte \(start)"
            }
        }
    }

    /// Streams `request` into `partURL`. When `partURL` already holds bytes, sends
    /// `Range: bytes=<n>-` and appends only a 206 whose Content-Range starts at exactly `n`.
    /// A 200 restarts the file from byte zero; any other partial response is rejected
    /// before a byte is appended. Cancelling the calling task leaves the written prefix in place.
    package static func stream(
        _ request: URLRequest,
        into partURL: URL,
        filePath: String,
        session: URLSession,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(
            at: partURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        var request = request
        let existingBytes = fileSize(partURL)
        if existingBytes > 0 {
            request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
        } else {
            request.setValue(nil, forHTTPHeaderField: "Range")
        }
        let delegate = Delegate(partURL: partURL, existingBytes: existingBytes,
                                filePath: filePath, progress: progress)
        let task = session.dataTask(with: request)
        task.delegate = delegate
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                delegate.continuation = continuation
                task.resume()
            }
        } onCancel: {
            // Ends the stream; the bytes already written stay in `.part` for next time.
            task.cancel()
        }
    }

    package static func fileSize(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
    }

    /// Start offset of a `Content-Range: bytes <start>-<end>/<total>` header, or nil.
    static func contentRangeStart(_ header: String?) -> Int64? {
        guard let header else { return nil }
        let trimmed = header.trimmingCharacters(in: .whitespaces)
        guard trimmed.lowercased().hasPrefix("bytes ") else { return nil }
        let spec = trimmed.dropFirst("bytes ".count)
        guard let dash = spec.firstIndex(of: "-") else { return nil }
        return Int64(spec[spec.startIndex..<dash].trimmingCharacters(in: .whitespaces))
    }

    /// Per-task delegate that writes the body into `.part` as it arrives. The handle and the
    /// continuation are each closed/resumed exactly once, guarded by `lock`.
    private final class Delegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private let partURL: URL
        private let existingBytes: Int64
        private let filePath: String
        private let progress: @Sendable (Int64) -> Void

        private let lock = NSLock()
        private var handle: FileHandle?
        private var writtenTotal: Int64
        private var finished = false
        private var setupError: Error?

        private var lastProgressTime = DispatchTime.now()
        private var bytesSinceProgress: Int64 = 0

        var continuation: CheckedContinuation<Void, Error>?

        init(partURL: URL, existingBytes: Int64, filePath: String,
             progress: @escaping @Sendable (Int64) -> Void) {
            self.partURL = partURL
            self.existingBytes = existingBytes
            self.filePath = filePath
            self.progress = progress
            self.writtenTotal = existingBytes
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive response: URLResponse,
                        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
            guard let http = response as? HTTPURLResponse else {
                fail(with: Failure.notHTTP(filePath))
                completionHandler(.cancel)
                return
            }
            guard http.statusCode == 200 || http.statusCode == 206 else {
                fail(with: Failure.badStatus(filePath, http.statusCode))
                completionHandler(.cancel)
                return
            }

            let append: Bool
            if http.statusCode == 206 {
                let header = http.value(forHTTPHeaderField: "Content-Range")
                guard existingBytes > 0, ResumableFileDownload.contentRangeStart(header) == existingBytes else {
                    // Appending a partial body at the wrong offset would corrupt the file. Drop the
                    // partial so the next attempt starts clean instead of repeating the same Range.
                    try? FileManager.default.removeItem(at: partURL)
                    fail(with: Failure.rangeMismatch(filePath, expectedStart: existingBytes, header: header))
                    completionHandler(.cancel)
                    return
                }
                append = true
            } else {
                // 200: the whole file follows, whether or not a Range was sent. Start over.
                append = false
            }

            do {
                let h: FileHandle
                if append {
                    h = try FileHandle(forWritingTo: partURL)
                    try h.seekToEnd()
                } else {
                    if FileManager.default.fileExists(atPath: partURL.path) {
                        try FileManager.default.removeItem(at: partURL)
                    }
                    FileManager.default.createFile(atPath: partURL.path, contents: nil)
                    h = try FileHandle(forWritingTo: partURL)
                }
                lock.lock(); handle = h; writtenTotal = append ? existingBytes : 0; lock.unlock()
            } catch {
                fail(with: error)
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            lock.lock()
            guard let h = handle, !finished else { lock.unlock(); return }
            do {
                try h.write(contentsOf: data)
                writtenTotal += Int64(data.count)
                bytesSinceProgress += Int64(data.count)
                let now = DispatchTime.now()
                let elapsedMs = (now.uptimeNanoseconds - lastProgressTime.uptimeNanoseconds) / 1_000_000
                let total = writtenTotal
                let shouldReport = elapsedMs >= 250 || bytesSinceProgress >= 8 * 1024 * 1024
                if shouldReport {
                    lastProgressTime = now
                    bytesSinceProgress = 0
                }
                lock.unlock()
                if shouldReport { progress(total) }
            } catch {
                lock.unlock()
                fail(with: error)
                dataTask.cancel()
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            lock.lock()
            if finished { lock.unlock(); return }
            finished = true
            try? handle?.close()
            handle = nil
            let pending = continuation
            continuation = nil
            let setup = setupError
            let total = writtenTotal
            lock.unlock()

            progress(total)

            if let setup {
                pending?.resume(throwing: setup)
            } else if let error {
                pending?.resume(throwing: error)
            } else {
                pending?.resume(returning: ())
            }
        }

        private func fail(with error: Error) {
            lock.lock()
            if setupError == nil { setupError = error }
            lock.unlock()
        }
    }
}
