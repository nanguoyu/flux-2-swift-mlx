// ModelDownloaderLayoutTests.swift - Download layout, completeness and resume over a stubbed hub
//
// No real network: every request goes to `StubHub` through the downloader's session configuration.

import CryptoKit
import XCTest
@testable import Flux2Core

/// Serves a fake HuggingFace tree listing plus file bodies, honoring `Range`.
final class StubHub: URLProtocol {
    struct Response { var status: Int; var headers: [String: String]; var body: Data }

    nonisolated(unsafe) private static var files: [String: Data] = [:]      // "repo/path" -> body
    nonisolated(unsafe) private static var listings: [String: [[String: Any]]] = [:]
    nonisolated(unsafe) static var overrides: [String: (URLRequest) -> Response] = [:]
    nonisolated(unsafe) static var requests: [URLRequest] = []
    private static let lock = NSLock()

    static func reset() {
        lock.lock(); defer { lock.unlock() }
        files = [:]; listings = [:]; overrides = [:]; requests = []
    }

    /// Publish `repo` with `files` (path -> body). Bodies get LFS size + sha256 like the real API.
    static func publish(repo: String, files repoFiles: [String: Data]) {
        lock.lock(); defer { lock.unlock() }
        var listing: [[String: Any]] = []
        for (path, body) in repoFiles.sorted(by: { $0.key < $1.key }) {
            files["\(repo)/\(path)"] = body
            let sha = SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined()
            listing.append(["type": "file", "path": path, "size": body.count,
                            "lfs": ["oid": sha, "size": body.count]])
        }
        listings[repo] = listing
    }

    static var recordedPaths: [String] {
        lock.lock(); defer { lock.unlock() }
        return requests.compactMap { $0.url?.path }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let url = request.url!
        let response = Self.respond(to: request)
        let http = HTTPURLResponse(url: url, statusCode: response.status, httpVersion: "HTTP/1.1",
                                   headerFields: response.headers)!
        client?.urlProtocol(self, didReceive: http, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: response.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    private static func respond(to request: URLRequest) -> Response {
        lock.lock()
        requests.append(request)
        let path = request.url!.path.removingPercentEncoding ?? request.url!.path
        let override = overrides[path]
        let listingRepo = path.hasPrefix("/api/models/")
            ? String(path.dropFirst("/api/models/".count).replacingOccurrences(of: "/tree/main", with: ""))
            : nil
        let listing = listingRepo.flatMap { listings[$0] }
        var body: Data?
        if let range = path.range(of: "/resolve/main/") {
            let repo = String(path[path.index(after: path.startIndex)..<range.lowerBound])
            body = files["\(repo)/\(path[range.upperBound...])"]
        }
        lock.unlock()

        if let override { return override(request) }
        if let listing {
            return Response(status: 200, headers: ["Content-Type": "application/json"],
                            body: try! JSONSerialization.data(withJSONObject: listing))
        }
        guard let body else { return Response(status: 404, headers: [:], body: Data()) }
        if let range = request.value(forHTTPHeaderField: "Range"),
           range.hasPrefix("bytes="), let start = Int(range.dropFirst(6).dropLast()), start < body.count {
            return Response(status: 206,
                            headers: ["Content-Range": "bytes \(start)-\(body.count - 1)/\(body.count)"],
                            body: body.subdata(in: start..<body.count))
        }
        return Response(status: 200, headers: [:], body: body)
    }

    static func configuration() -> URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubHub.self]
        return config
    }
}

final class ModelDownloaderLayoutTests: XCTestCase {
    private var root: URL!

    // A Klein-shaped repository: component weights at the root, plus other components' folders
    // whose file names collide with the root's when flattened.
    private let int8Repo = ModelRegistry.TransformerVariant.klein4B_8bit.huggingFaceRepo
    private let transformerBody = Data(repeating: 0xA1, count: 4096)
    private var int8Files: [String: Data] {
        [
            "config.json": Data(#"{"component":"transformer"}"#.utf8),
            "diffusion_pytorch_model.safetensors": transformerBody,
            "quanto_qmap.json": Data(#"{"q":1}"#.utf8),
            "README.md": Data("readme".utf8),
            "text_encoder/config.json": Data(#"{"component":"text_encoder"}"#.utf8),
            "text_encoder/model.safetensors": Data(repeating: 0xB2, count: 3000),
            "tokenizer/tokenizer.json": Data(#"{"t":1}"#.utf8),
            "vae/config.json": Data(#"{"component":"vae"}"#.utf8),
            "vae/diffusion_pytorch_model.safetensors": Data(repeating: 0xC3, count: 1000),
        ]
    }

    override func setUpWithError() throws {
        StubHub.reset()
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-downloader-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ModelRegistry.customModelsDirectory = root
    }

    override func tearDownWithError() throws {
        ModelRegistry.customModelsDirectory = nil
        StubHub.reset()
        try? FileManager.default.removeItem(at: root)
    }

    private func downloader() -> Flux2ModelDownloader {
        Flux2ModelDownloader(sessionConfiguration: StubHub.configuration())
    }

    private var int8Dir: URL { ModelRegistry.localPath(for: .transformer(.klein4B_8bit)) }

    private func topLevelNames(_ dir: URL) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
    }

    // MARK: - Layout

    func testRootComponentDownloadsOnlyItsOwnTopLevelFiles() async throws {
        StubHub.publish(repo: int8Repo, files: int8Files)

        let dir = try await downloader().download(.transformer(.klein4B_8bit))

        XCTAssertEqual(dir.standardizedFileURL, int8Dir.standardizedFileURL)
        XCTAssertEqual(topLevelNames(dir).subtracting([".mobile-diffuser-download-manifest.json"]),
                       ["config.json", "diffusion_pytorch_model.safetensors", "quanto_qmap.json"])
        // The transformer weights must not be overwritten by vae/diffusion_pytorch_model.safetensors.
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("diffusion_pytorch_model.safetensors")),
                       transformerBody)
        XCTAssertEqual(try String(contentsOf: dir.appendingPathComponent("config.json"), encoding: .utf8),
                       #"{"component":"transformer"}"#)
        XCTAssertFalse(StubHub.recordedPaths.contains { $0.contains("/text_encoder/") || $0.contains("/vae/") },
                       "files of other components must not be fetched")
        XCTAssertNotNil(Flux2ModelDownloader.findModelPath(for: .transformer(.klein4B_8bit)))
    }

    func testSubfolderComponentKeepsOnlyDirectChildren() {
        let files: [Flux2ModelDownloader.HubFile] = [
            .init(path: "transformer/0.safetensors", size: 1, sha256: nil),
            .init(path: "transformer/model.safetensors.index.json", size: 1, sha256: nil),
            .init(path: "transformer/nested/config.json", size: 1, sha256: nil),
            .init(path: "vae/config.json", size: 1, sha256: nil),
            .init(path: "config.json", size: 1, sha256: nil),
        ]
        let kept = Flux2ModelDownloader.componentFiles(from: files, subfolder: "transformer").map(\.path)
        XCTAssertEqual(kept, ["transformer/0.safetensors", "transformer/model.safetensors.index.json"])

        let rootKept = Flux2ModelDownloader.componentFiles(from: files, subfolder: nil).map(\.path)
        XCTAssertEqual(rootKept, ["config.json"])
    }

    // MARK: - Existing layouts

    func testLegacyLayoutWithoutManifestStaysInstalledAndIsNotRefetched() async throws {
        let dir = int8Dir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        try transformerBody.write(to: dir.appendingPathComponent("diffusion_pytorch_model.safetensors"))

        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.transformer(.klein4B_8bit)))
        _ = try await downloader().download(.transformer(.klein4B_8bit))
        XCTAssertTrue(StubHub.recordedPaths.isEmpty, "an installed legacy layout must not touch the network")
    }

    func testCollidedRecursiveLayoutIsRepairedByTheNextDownload() async throws {
        // What the recursive listing produced: VAE weights over the transformer file, the text
        // encoder beside it, and a manifest listing the same names twice with different sizes.
        let dir = int8Dir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(#"{"component":"vae"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data(repeating: 0xC3, count: 1000).write(to: dir.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        try Data(repeating: 0xB2, count: 3000).write(to: dir.appendingPathComponent("model.safetensors"))
        try Data(#"{"t":1}"#.utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        let manifest: [String: Any] = ["version": 1, "files": [
            ["filename": "config.json", "size": 27],
            ["filename": "diffusion_pytorch_model.safetensors", "size": 4096],
            ["filename": "model.safetensors", "size": 3000],
            ["filename": "tokenizer.json", "size": 7],
            ["filename": "config.json", "size": 19],
            ["filename": "diffusion_pytorch_model.safetensors", "size": 1000],
        ]]
        try JSONSerialization.data(withJSONObject: manifest)
            .write(to: dir.appendingPathComponent(".mobile-diffuser-download-manifest.json"))

        XCTAssertFalse(Flux2ModelDownloader.isDownloaded(.transformer(.klein4B_8bit)))

        StubHub.publish(repo: int8Repo, files: int8Files)
        _ = try await downloader().download(.transformer(.klein4B_8bit))

        XCTAssertEqual(topLevelNames(dir).subtracting([".mobile-diffuser-download-manifest.json"]),
                       ["config.json", "diffusion_pytorch_model.safetensors", "quanto_qmap.json"])
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("diffusion_pytorch_model.safetensors")),
                       transformerBody)
        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.transformer(.klein4B_8bit)))
    }

    // MARK: - Interruption and resume

    func testInterruptedFirstDownloadIsNotReportedInstalled() async throws {
        StubHub.publish(repo: int8Repo, files: int8Files)
        // The weights arrive; the last file of the component fails.
        StubHub.overrides["/\(int8Repo)/resolve/main/quanto_qmap.json"] = { _ in
            .init(status: 503, headers: [:], body: Data())
        }

        do {
            _ = try await downloader().download(.transformer(.klein4B_8bit))
            XCTFail("download should fail")
        } catch {}

        XCTAssertTrue(FileManager.default.fileExists(
            atPath: int8Dir.appendingPathComponent("diffusion_pytorch_model.safetensors").path))
        XCTAssertFalse(Flux2ModelDownloader.isDownloaded(.transformer(.klein4B_8bit)),
                       "config + weights without the rest of the component must not read as installed")
    }

    func testPartialFileResumesFromTheRetainedByteCount() async throws {
        StubHub.publish(repo: int8Repo, files: int8Files)
        let dir = int8Dir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try transformerBody.prefix(1000)
            .write(to: dir.appendingPathComponent("diffusion_pytorch_model.safetensors.part"))

        _ = try await downloader().download(.transformer(.klein4B_8bit))

        let ranged = StubHub.requests.first { $0.url?.path.hasSuffix("diffusion_pytorch_model.safetensors") == true }
        XCTAssertEqual(ranged?.value(forHTTPHeaderField: "Range"), "bytes=1000-")
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("diffusion_pytorch_model.safetensors")),
                       transformerBody)
    }

    func testMisplacedPartialResponseIsRejectedBeforeAppend() async throws {
        StubHub.publish(repo: int8Repo, files: int8Files)
        let dir = int8Dir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let part = dir.appendingPathComponent("diffusion_pytorch_model.safetensors.part")
        try transformerBody.prefix(1000).write(to: part)
        let body = transformerBody
        StubHub.overrides["/\(int8Repo)/resolve/main/diffusion_pytorch_model.safetensors"] = { _ in
            .init(status: 206, headers: ["Content-Range": "bytes 0-\(body.count - 1)/\(body.count)"], body: body)
        }

        do {
            _ = try await downloader().download(.transformer(.klein4B_8bit))
            XCTFail("a 206 that does not continue at the retained byte count must fail")
        } catch {}
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("diffusion_pytorch_model.safetensors").path))
    }
}
