// Qwen3DownloadResumeTests.swift - Qwen3 text-encoder download over a stubbed hub (no network)

import CryptoKit
import XCTest
@testable import FluxTextEncoders

/// Fake HuggingFace tree listing + file bodies for one repo, honoring `Range`.
final class QwenStubHub: URLProtocol {
    nonisolated(unsafe) static var repo = ""
    /// The only revision served; listing and resolve requests for any other get 404.
    nonisolated(unsafe) static var revision = "main"
    nonisolated(unsafe) static var files: [String: Data] = [:]
    nonisolated(unsafe) static var failing: Set<String> = []
    nonisolated(unsafe) static var requests: [URLRequest] = []
    private static let lock = NSLock()

    static func reset() {
        lock.lock(); defer { lock.unlock() }
        repo = ""; revision = "main"; files = [:]; failing = []; requests = []
    }

    /// A pin naming `names` of the published files at the served revision.
    static func pin(_ names: [String]) -> HubSnapshotPin {
        HubSnapshotPin(revision: revision, files: names.map { name in
            let data = files[name]!
            let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            return HubPinnedFile(name, size: Int64(data.count), sha256: sha)
        })
    }

    static func etag(_ data: Data) -> String {
        "\"" + SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() + "\""
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        Self.lock.lock()
        Self.requests.append(request)
        let path = request.url!.path.removingPercentEncoding ?? request.url!.path
        let repo = Self.repo, revision = Self.revision, files = Self.files, failing = Self.failing
        Self.lock.unlock()

        var status = 404
        var headers: [String: String] = [:]
        var body = Data()
        if path == "/api/models/\(repo)/tree/\(revision)" {
            let listing: [[String: Any]] = files.keys.sorted().map { name in
                let data = files[name]!
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                return ["type": "file", "path": name, "size": data.count, "lfs": ["oid": sha, "size": data.count]]
            }
            status = 200
            body = try! JSONSerialization.data(withJSONObject: listing)
        } else if path.hasPrefix("/\(repo)/resolve/\(revision)/") {
            let name = String(path.dropFirst("/\(repo)/resolve/\(revision)/".count))
            if failing.contains(name) {
                status = 503
            } else if let data = files[name] {
                let etag = Self.etag(data)
                headers["ETag"] = etag
                let ifRange = request.value(forHTTPHeaderField: "If-Range")
                if let range = request.value(forHTTPHeaderField: "Range"), ifRange == nil || ifRange == etag,
                   let start = Int(range.dropFirst("bytes=".count).dropLast()), start < data.count {
                    status = 206
                    headers["Content-Range"] = "bytes \(start)-\(data.count - 1)/\(data.count)"
                    body = data.subdata(in: start..<data.count)
                } else {
                    status = 200
                    body = data
                }
            }
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                                       headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
}

final class Qwen3DownloadResumeTests: XCTestCase {
    // A variant that is not in a developer's ~/.cache/huggingface, so only our temp root counts.
    private let variant = Qwen3Variant.qwen3_8B_4bit
    private let weights = Data((0..<5000).map { UInt8($0 % 251) })
    private var root: URL!

    override func setUpWithError() throws {
        QwenStubHub.reset()
        QwenStubHub.repo = variant.repoId
        QwenStubHub.files = [
            "config.json": Data(#"{"model_type":"qwen3"}"#.utf8),
            "model.safetensors": weights,
            "tokenizer.json": Data(#"{"t":1}"#.utf8),
            "tokenizer_config.json": Data(#"{"c":1}"#.utf8),
            "nested/config.json": Data(#"{"other":1}"#.utf8),
        ]
        // Pinned to a fixed commit, like the shipped variants: the top-level files only.
        QwenStubHub.revision = "89abcdef0123456789abcdef0123456789abcdef"
        let pin = QwenStubHub.pin(["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])
        TextEncoderModelDownloader.qwen3PinOverrideForTesting = { _ in pin }
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-downloader-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        TextEncoderModelDownloader.customModelsDirectory = root
    }

    override func tearDownWithError() throws {
        TextEncoderModelDownloader.qwen3PinOverrideForTesting = nil
        TextEncoderModelDownloader.customModelsDirectory = nil
        QwenStubHub.reset()
        try? FileManager.default.removeItem(at: root)
    }

    private var repoDir: URL {
        variant.repoId.split(separator: "/").reduce(root) { $0.appendingPathComponent(String($1)) }
    }

    private func downloader() -> TextEncoderModelDownloader {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [QwenStubHub.self]
        return TextEncoderModelDownloader(sessionConfiguration: config)
    }

    func testCompleteDownloadIsFlatAndInstalled() async throws {
        _ = try await downloader().downloadQwen3(variant: variant)

        XCTAssertEqual(try Data(contentsOf: repoDir.appendingPathComponent("model.safetensors")), weights)
        XCTAssertEqual(try String(contentsOf: repoDir.appendingPathComponent("config.json"), encoding: .utf8),
                       #"{"model_type":"qwen3"}"#, "a nested config.json must not replace the root one")
        XCTAssertTrue(TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: variant))
    }

    func testInterruptedAfterWeightsIsNotInstalled() async throws {
        QwenStubHub.failing = ["tokenizer_config.json"]
        do {
            _ = try await downloader().downloadQwen3(variant: variant)
            XCTFail("download should fail")
        } catch {}

        XCTAssertTrue(FileManager.default.fileExists(atPath: repoDir.appendingPathComponent("model.safetensors").path))
        XCTAssertFalse(TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: variant))
    }

    func testPartialWeightsResumeWithRange() async throws {
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        let part = repoDir.appendingPathComponent("model.safetensors.part")
        try weights.prefix(1234).write(to: part)
        try ResumableFileDownload.writeValidator(.init(etag: QwenStubHub.etag(weights)), for: part)

        _ = try await downloader().downloadQwen3(variant: variant)

        let request = QwenStubHub.requests.first { $0.url?.path.hasSuffix("/model.safetensors") == true }
        XCTAssertEqual(request?.value(forHTTPHeaderField: "Range"), "bytes=1234-")
        XCTAssertEqual(request?.value(forHTTPHeaderField: "If-Range"), QwenStubHub.etag(weights))
        XCTAssertEqual(try Data(contentsOf: repoDir.appendingPathComponent("model.safetensors")), weights)
        XCTAssertFalse(FileManager.default.fileExists(atPath: part.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: ResumableFileDownload.validatorURL(for: part).path))
    }

    func testChangedUpstreamFileRestartsFromZero() async throws {
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        let part = repoDir.appendingPathComponent("model.safetensors.part")
        try Data(repeating: 0xEE, count: 1234).write(to: part)
        try ResumableFileDownload.writeValidator(.init(etag: "\"old-upload\""), for: part)

        _ = try await downloader().downloadQwen3(variant: variant)

        let request = QwenStubHub.requests.first { $0.url?.path.hasSuffix("/model.safetensors") == true }
        XCTAssertEqual(request?.value(forHTTPHeaderField: "If-Range"), "\"old-upload\"")
        XCTAssertEqual(try Data(contentsOf: repoDir.appendingPathComponent("model.safetensors")), weights)
    }

    /// The validator recorded for a later resume is the FINAL response's, not the redirect's.
    func testValidatorComesFromTheFinalResponseAfterRedirect() async throws {
        let part = root.appendingPathComponent("blob.part")
        var request = URLRequest(url: URL(string: "https://huggingface.co/\(variant.repoId)/resolve/main/redirected.bin")!)
        request.setValue("Bearer x", forHTTPHeaderField: "Authorization")
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RedirectingStub.self]
        RedirectingStub.body = Data(repeating: 7, count: 300)

        try await ResumableFileDownload.stream(request, into: part, filePath: "redirected.bin",
                                               expectedSize: 300, session: URLSession(configuration: config),
                                               progress: { _ in })

        XCTAssertEqual(ResumableFileDownload.readValidator(for: part), .init(etag: "\"final\""))
        try ResumableFileDownload.commit(part, to: root.appendingPathComponent("blob"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: ResumableFileDownload.validatorURL(for: part).path))
    }

    func testWeakETagFallsBackToLastModified() {
        let url = URL(string: "https://example.invalid/f")!
        let weak = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil,
                                   headerFields: ["ETag": "W/\"abc\"", "Last-Modified": "Tue, 10 Nov 2026 10:00:00 GMT"])!
        XCTAssertEqual(ResumableFileDownload.Validator(response: weak),
                       .init(lastModified: "Tue, 10 Nov 2026 10:00:00 GMT"))
        let none = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: [:])!
        XCTAssertNil(ResumableFileDownload.Validator(response: none))
    }

    func testContentRangeParsing() {
        XCTAssertEqual(ResumableFileDownload.contentRangeStart("bytes 1234-4999/5000"), 1234)
        XCTAssertEqual(ResumableFileDownload.contentRangeTotal("bytes 1234-4999/5000"), 5000)
        XCTAssertNil(ResumableFileDownload.contentRangeTotal("bytes 1234-4999/*"))
        XCTAssertNil(ResumableFileDownload.contentRangeStart("bytes */5000"))
        XCTAssertNil(ResumableFileDownload.contentRangeStart(nil))
    }
}

/// Redirects every `huggingface.co` request to a CDN host; the redirect and the CDN answer carry
/// different ETags.
final class RedirectingStub: URLProtocol {
    nonisolated(unsafe) static var body = Data()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let url = request.url!
        if url.host == "huggingface.co" {
            let target = URL(string: "https://cdn.stub.invalid/blob")!
            let redirect = HTTPURLResponse(url: url, statusCode: 302, httpVersion: "HTTP/1.1",
                                           headerFields: ["Location": target.absoluteString, "ETag": "\"redirect\""])!
            client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: target), redirectResponse: redirect)
            return
        }
        let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                       headerFields: ["ETag": "\"final\""])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.body)
        client?.urlProtocolDidFinishLoading(self)
    }
}
