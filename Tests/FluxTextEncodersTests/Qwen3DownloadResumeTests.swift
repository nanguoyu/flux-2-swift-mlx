// Qwen3DownloadResumeTests.swift - Qwen3 text-encoder download over a stubbed hub (no network)

import CryptoKit
import XCTest
@testable import FluxTextEncoders

/// Fake HuggingFace tree listing + file bodies for one repo, honoring `Range`.
final class QwenStubHub: URLProtocol {
    nonisolated(unsafe) static var repo = ""
    nonisolated(unsafe) static var files: [String: Data] = [:]
    nonisolated(unsafe) static var failing: Set<String> = []
    nonisolated(unsafe) static var requests: [URLRequest] = []
    private static let lock = NSLock()

    static func reset() {
        lock.lock(); defer { lock.unlock() }
        repo = ""; files = [:]; failing = []; requests = []
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        Self.lock.lock()
        Self.requests.append(request)
        let path = request.url!.path.removingPercentEncoding ?? request.url!.path
        let repo = Self.repo, files = Self.files, failing = Self.failing
        Self.lock.unlock()

        var status = 404
        var headers: [String: String] = [:]
        var body = Data()
        if path == "/api/models/\(repo)/tree/main" {
            let listing: [[String: Any]] = files.keys.sorted().map { name in
                let data = files[name]!
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                return ["type": "file", "path": name, "size": data.count, "lfs": ["oid": sha, "size": data.count]]
            }
            status = 200
            body = try! JSONSerialization.data(withJSONObject: listing)
        } else if path.hasPrefix("/\(repo)/resolve/main/") {
            let name = String(path.dropFirst("/\(repo)/resolve/main/".count))
            if failing.contains(name) {
                status = 503
            } else if let data = files[name] {
                if let range = request.value(forHTTPHeaderField: "Range"),
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
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-downloader-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        TextEncoderModelDownloader.customModelsDirectory = root
    }

    override func tearDownWithError() throws {
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
        try weights.prefix(1234).write(to: repoDir.appendingPathComponent("model.safetensors.part"))

        _ = try await downloader().downloadQwen3(variant: variant)

        let request = QwenStubHub.requests.first { $0.url?.path.hasSuffix("/model.safetensors") == true }
        XCTAssertEqual(request?.value(forHTTPHeaderField: "Range"), "bytes=1234-")
        XCTAssertEqual(try Data(contentsOf: repoDir.appendingPathComponent("model.safetensors")), weights)
        XCTAssertFalse(FileManager.default.fileExists(atPath: repoDir.appendingPathComponent("model.safetensors.part").path))
    }

    func testContentRangeStartParsing() {
        XCTAssertEqual(ResumableFileDownload.contentRangeStart("bytes 1234-4999/5000"), 1234)
        XCTAssertNil(ResumableFileDownload.contentRangeStart("bytes */5000"))
        XCTAssertNil(ResumableFileDownload.contentRangeStart(nil))
    }
}
