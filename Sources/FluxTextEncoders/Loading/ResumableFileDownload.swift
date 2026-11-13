// ResumableFileDownload.swift - Stream one HTTP body into a resumable `.part` file
//
// Shared by Flux2ModelDownloader (Flux2Core) and TextEncoderModelDownloader (this module) so both
// the diffusion components and the Qwen3 text encoder keep every received byte across a cancel,
// pause or network drop.
//
// A `.part` is only resumed when the response that started it left a validator in the sidecar
// `<name>.part.validator` (its strong ETag, else its Last-Modified). The validator is sent back as
// `If-Range`, so a file replaced upstream between attempts comes back as a full 200 instead of
// being spliced onto the old prefix.

import Foundation

package enum ResumableFileDownload {
    package enum Failure: Error, LocalizedError {
        case notHTTP(String)
        case badStatus(String, Int)
        case rangeMismatch(String, expectedStart: Int64, header: String?)
        case totalMismatch(String, expected: Int64, header: String?)
        case validatorMismatch(String)
        case sizeMismatch(String, expected: Int64, actual: Int64)

        package var errorDescription: String? {
            switch self {
            case .notHTTP(let path):
                return "Failed to download \(path)"
            case .badStatus(let path, let code):
                return "Failed to download \(path): HTTP \(code)"
            case .rangeMismatch(let path, let start, let header):
                return "Failed to download \(path): partial response \(header ?? "without Content-Range") does not continue at byte \(start)"
            case .totalMismatch(let path, let expected, let header):
                return "Failed to download \(path): partial response \(header ?? "without Content-Range") is not part of a \(expected)-byte file"
            case .validatorMismatch(let path):
                return "Failed to download \(path): the file changed on the server since the download started"
            case .sizeMismatch(let path, let expected, let actual):
                return "Failed to download \(path): received \(actual) bytes, expected \(expected)"
            }
        }
    }

    /// What identifies the exact upstream file a `.part` holds a prefix of.
    struct Validator: Codable, Equatable {
        var etag: String?
        var lastModified: String?

        /// The strong ETag of `response`, else its Last-Modified; nil when it has neither.
        /// A weak ETag (`W/"…"`) is not byte-exact, so it cannot validate a range.
        init?(response: HTTPURLResponse) {
            if let etag = response.value(forHTTPHeaderField: "ETag")?.trimmingCharacters(in: .whitespaces),
               !etag.isEmpty, !etag.hasPrefix("W/") {
                self.etag = etag
            } else if let date = response.value(forHTTPHeaderField: "Last-Modified")?
                        .trimmingCharacters(in: .whitespaces), !date.isEmpty {
                self.lastModified = date
            } else {
                return nil
            }
        }

        init(etag: String? = nil, lastModified: String? = nil) {
            self.etag = etag
            self.lastModified = lastModified
        }

        /// The `If-Range` value for this validator.
        var ifRange: String? { etag ?? lastModified }

        /// True when `response` carries a validator of the same kind that names a different file.
        func contradicts(_ response: HTTPURLResponse) -> Bool {
            if let etag {
                guard let other = response.value(forHTTPHeaderField: "ETag")?.trimmingCharacters(in: .whitespaces),
                      !other.isEmpty, !other.hasPrefix("W/") else { return false }
                return other != etag
            }
            if let lastModified,
               let other = response.value(forHTTPHeaderField: "Last-Modified")?.trimmingCharacters(in: .whitespaces),
               !other.isEmpty {
                return other != lastModified
            }
            return false
        }
    }

    /// Streams `request` into `partURL`.
    ///
    /// - A retained `.part` is resumed only when its validator sidecar exists: the request then
    ///   carries `Range: bytes=<n>-` and `If-Range: <validator>`. Without a validator, or when the
    ///   retained bytes already reach `expectedSize`, the partial is discarded and the download
    ///   starts at byte zero.
    /// - A 206 is appended only when its Content-Range starts at exactly `n`, its total equals
    ///   `expectedSize` (when known) and it does not carry a different validator. Anything else is
    ///   rejected before a byte is appended, and the partial plus its sidecar are removed.
    /// - A 200 restarts the file from byte zero and records that response's validator.
    /// - Redirects (HuggingFace sends LFS files to a CDN) keep `Range` and `If-Range`; the
    ///   validator is always taken from the final response.
    /// - After the body completes, a known `expectedSize` must match the `.part` exactly.
    ///
    /// Cancelling the calling task leaves the written prefix and its sidecar in place. On success
    /// the `.part` and sidecar remain for the caller to verify and `commit`.
    package static func stream(
        _ request: URLRequest,
        into partURL: URL,
        filePath: String,
        expectedSize: Int64? = nil,
        session: URLSession,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(
            at: partURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        var existingBytes = fileSize(partURL)
        let stored = readValidator(for: partURL)
        if existingBytes > 0 {
            let alreadyFull = expectedSize.map { existingBytes >= $0 } ?? false
            if stored?.ifRange == nil || alreadyFull {
                discardPartial(partURL)
                existingBytes = 0
            }
        } else {
            discardPartial(partURL)
        }

        var request = request
        let range: String?
        let ifRange: String?
        if existingBytes > 0, let value = stored?.ifRange {
            range = "bytes=\(existingBytes)-"
            ifRange = value
        } else {
            range = nil
            ifRange = nil
        }
        request.setValue(range, forHTTPHeaderField: "Range")
        request.setValue(ifRange, forHTTPHeaderField: "If-Range")

        let delegate = Delegate(partURL: partURL, existingBytes: existingBytes,
                                storedValidator: existingBytes > 0 ? stored : nil,
                                range: range, ifRange: ifRange, expectedSize: expectedSize,
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

        if let expectedSize {
            let actual = fileSize(partURL)
            guard actual == expectedSize else {
                discardPartial(partURL)
                throw Failure.sizeMismatch(filePath, expected: expectedSize, actual: actual)
            }
        }
    }

    /// The sidecar that records which upstream file `partURL` holds a prefix of.
    package static func validatorURL(for partURL: URL) -> URL {
        partURL.appendingPathExtension("validator")
    }

    /// Removes a partial download and its validator sidecar.
    package static func discardPartial(_ partURL: URL) {
        try? FileManager.default.removeItem(at: partURL)
        try? FileManager.default.removeItem(at: validatorURL(for: partURL))
    }

    /// Moves a verified `.part` into place and removes its validator sidecar.
    package static func commit(_ partURL: URL, to destination: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.moveItem(at: partURL, to: destination)
        try? fm.removeItem(at: validatorURL(for: partURL))
    }

    package static func fileSize(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
    }

    static func readValidator(for partURL: URL) -> Validator? {
        guard let data = try? Data(contentsOf: validatorURL(for: partURL)) else { return nil }
        return try? JSONDecoder().decode(Validator.self, from: data)
    }

    static func writeValidator(_ validator: Validator?, for partURL: URL) throws {
        let url = validatorURL(for: partURL)
        guard let validator else {
            try? FileManager.default.removeItem(at: url)
            return
        }
        try JSONEncoder().encode(validator).write(to: url, options: .atomic)
    }

    /// Start offset of a `Content-Range: bytes <start>-<end>/<total>` header, or nil.
    static func contentRangeStart(_ header: String?) -> Int64? {
        guard let spec = rangeSpec(header), let dash = spec.firstIndex(of: "-") else { return nil }
        return Int64(spec[spec.startIndex..<dash].trimmingCharacters(in: .whitespaces))
    }

    /// Complete length of a `Content-Range: bytes <start>-<end>/<total>` header; nil when it is
    /// missing or `*`.
    static func contentRangeTotal(_ header: String?) -> Int64? {
        guard let spec = rangeSpec(header), let slash = spec.lastIndex(of: "/") else { return nil }
        return Int64(spec[spec.index(after: slash)...].trimmingCharacters(in: .whitespaces))
    }

    private static func rangeSpec(_ header: String?) -> Substring? {
        guard let header else { return nil }
        let trimmed = header.trimmingCharacters(in: .whitespaces)
        guard trimmed.lowercased().hasPrefix("bytes ") else { return nil }
        return Substring(trimmed.dropFirst("bytes ".count))
    }

    /// Per-task delegate that writes the body into `.part` as it arrives. The handle and the
    /// continuation are each closed/resumed exactly once, guarded by `lock`.
    private final class Delegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private let partURL: URL
        private let existingBytes: Int64
        private let storedValidator: Validator?
        private let range: String?
        private let ifRange: String?
        private let expectedSize: Int64?
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

        init(partURL: URL, existingBytes: Int64, storedValidator: Validator?,
             range: String?, ifRange: String?, expectedSize: Int64?, filePath: String,
             progress: @escaping @Sendable (Int64) -> Void) {
            self.partURL = partURL
            self.existingBytes = existingBytes
            self.storedValidator = storedValidator
            self.range = range
            self.ifRange = ifRange
            self.expectedSize = expectedSize
            self.filePath = filePath
            self.progress = progress
            self.writtenTotal = existingBytes
        }

        /// Keeps the resume headers on the redirected request (HuggingFace `resolve` URLs redirect
        /// LFS files to a CDN), so the CDN evaluates the same Range against the same validator.
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest,
                        completionHandler: @escaping (URLRequest?) -> Void) {
            var redirected = request
            redirected.setValue(range, forHTTPHeaderField: "Range")
            redirected.setValue(ifRange, forHTTPHeaderField: "If-Range")
            completionHandler(redirected)
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
                if http.statusCode == 416 {
                    // The retained prefix no longer fits the file; start clean next time.
                    ResumableFileDownload.discardPartial(partURL)
                }
                fail(with: Failure.badStatus(filePath, http.statusCode))
                completionHandler(.cancel)
                return
            }

            let append: Bool
            if http.statusCode == 206 {
                let header = http.value(forHTTPHeaderField: "Content-Range")
                if let rejection = rejectionOfPartial(http, header: header) {
                    // Appending this body would corrupt the file. Drop the partial and its
                    // validator so the next attempt starts clean instead of repeating the request.
                    ResumableFileDownload.discardPartial(partURL)
                    fail(with: rejection)
                    completionHandler(.cancel)
                    return
                }
                append = true
            } else {
                // 200: the whole file follows, whether or not a Range was sent. Start over and
                // remember which file this is so a later attempt can resume it.
                append = false
            }

            do {
                let h: FileHandle
                if append {
                    h = try FileHandle(forWritingTo: partURL)
                    try h.seekToEnd()
                } else {
                    ResumableFileDownload.discardPartial(partURL)
                    try ResumableFileDownload.writeValidator(Validator(response: http), for: partURL)
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

        private func rejectionOfPartial(_ http: HTTPURLResponse, header: String?) -> Failure? {
            guard existingBytes > 0, let storedValidator,
                  ResumableFileDownload.contentRangeStart(header) == existingBytes else {
                return .rangeMismatch(filePath, expectedStart: existingBytes, header: header)
            }
            if let expectedSize, ResumableFileDownload.contentRangeTotal(header) != expectedSize {
                return .totalMismatch(filePath, expected: expectedSize, header: header)
            }
            if storedValidator.contradicts(http) {
                return .validatorMismatch(filePath)
            }
            return nil
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
