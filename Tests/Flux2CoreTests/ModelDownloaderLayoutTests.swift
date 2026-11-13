// ModelDownloaderLayoutTests.swift - Download layout, completeness and resume over a stubbed hub
//
// No real network: every request goes to `StubHub` through the downloader's session configuration.

import CryptoKit
import XCTest
import FluxTextEncoders
@testable import Flux2Core

/// Serves a fake HuggingFace tree listing plus file bodies, honoring `Range` and `If-Range`.
/// Every file body carries a strong ETag (its quoted sha256).
final class StubHub: URLProtocol {
    struct Response {
        var status: Int
        var headers: [String: String]
        var body: Data
        var redirect: URL? = nil
    }

    static func etag(_ body: Data) -> String {
        "\"" + SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined() + "\""
    }

    nonisolated(unsafe) private static var files: [String: Data] = [:]      // "repo/path" -> body
    nonisolated(unsafe) private static var listings: [String: [[String: Any]]] = [:]
    nonisolated(unsafe) private static var revisions: [String: String] = [:]  // repo -> served revision
    nonisolated(unsafe) static var overrides: [String: (URLRequest) -> Response] = [:]
    nonisolated(unsafe) static var requests: [URLRequest] = []
    private static let lock = NSLock()

    static func reset() {
        lock.lock(); defer { lock.unlock() }
        files = [:]; listings = [:]; revisions = [:]; overrides = [:]; requests = []
    }

    /// Publish `repo` at `revision` with `files` (path -> body). Bodies get LFS size + sha256 like
    /// the real API. Listing and resolve requests for any other revision get 404.
    static func publish(repo: String, revision: String = "main", files repoFiles: [String: Data]) {
        lock.lock(); defer { lock.unlock() }
        revisions[repo] = revision
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
        if let target = response.redirect {
            var headers = response.headers
            headers["Location"] = target.absoluteString
            let redirect = HTTPURLResponse(url: url, statusCode: response.status, httpVersion: "HTTP/1.1",
                                           headerFields: headers)!
            // A plain request: the session's redirect handling must carry the resume headers over.
            client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: target), redirectResponse: redirect)
            return
        }
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
        // /api/models/<org>/<name>/tree/<revision>[/<folder>]: only the published revision lists,
        // and a folder listing returns that folder's entries (with full paths) like the real API.
        var listing: [[String: Any]]?
        if path.hasPrefix("/api/models/") {
            let parts = path.dropFirst("/api/models/".count).split(separator: "/", omittingEmptySubsequences: false)
            if parts.count >= 4, parts[2] == "tree" {
                let repo = "\(parts[0])/\(parts[1])"
                let folder = parts.dropFirst(4).joined(separator: "/")
                if revisions[repo] == String(parts[3]), let all = listings[repo] {
                    listing = folder.isEmpty ? all : all.filter { ($0["path"] as! String).hasPrefix(folder + "/") }
                }
            }
        }
        // /<org>/<name>/resolve/<revision>/<path>
        var body: Data?
        let parts = path.dropFirst().split(separator: "/", maxSplits: 4, omittingEmptySubsequences: false)
        if parts.count == 5, parts[2] == "resolve" {
            let repo = "\(parts[0])/\(parts[1])"
            if revisions[repo] == String(parts[3]) { body = files["\(repo)/\(parts[4])"] }
        }
        lock.unlock()

        if let override { return override(request) }
        if let listing {
            return Response(status: 200, headers: ["Content-Type": "application/json"],
                            body: try! JSONSerialization.data(withJSONObject: listing))
        }
        guard let body else { return Response(status: 404, headers: [:], body: Data()) }
        return serve(body, to: request)
    }

    /// RFC 9110 range handling: a Range whose If-Range does not match the current ETag gets the
    /// whole body.
    static func serve(_ body: Data, to request: URLRequest, etag: String? = nil) -> Response {
        let etag = etag ?? Self.etag(body)
        let ifRange = request.value(forHTTPHeaderField: "If-Range")
        if let range = request.value(forHTTPHeaderField: "Range"),
           ifRange == nil || ifRange == etag,
           range.hasPrefix("bytes="), let start = Int(range.dropFirst(6).dropLast()), start < body.count {
            return Response(status: 206,
                            headers: ["Content-Range": "bytes \(start)-\(body.count - 1)/\(body.count)",
                                      "ETag": etag],
                            body: body.subdata(in: start..<body.count))
        }
        return Response(status: 200, headers: ["ETag": etag], body: body)
    }

    /// A pin naming `paths` of `files` (path -> body) at `revision`.
    static func pin(revision: String, files: [String: Data], paths: [String]) -> HubSnapshotPin {
        HubSnapshotPin(revision: revision, files: paths.map { path in
            let body = files[path]!
            let sha = SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined()
            return HubPinnedFile(path, size: Int64(body.count), sha256: sha)
        })
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

    /// The commit the stub serves; every test component is pinned to it.
    private let rev = "0123456789abcdef0123456789abcdef01234567"
    private let int8Paths = ["config.json", "diffusion_pytorch_model.safetensors", "quanto_qmap.json"]
    private var int8Pin: HubSnapshotPin { StubHub.pin(revision: rev, files: int8Files, paths: int8Paths) }

    override func setUpWithError() throws {
        StubHub.reset()
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-downloader-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ModelRegistry.customModelsDirectory = root
        let pin = int8Pin
        Flux2ModelDownloader.pinOverrideForTesting = { _ in pin }
    }

    override func tearDownWithError() throws {
        Flux2ModelDownloader.pinOverrideForTesting = nil
        HubDownloadPolicy.allowsUnpinnedRevisions = false
        ModelRegistry.customModelsDirectory = nil
        StubHub.reset()
        try? FileManager.default.removeItem(at: root)
    }

    private func downloader() -> Flux2ModelDownloader {
        Flux2ModelDownloader(sessionConfiguration: StubHub.configuration())
    }

    private var int8Dir: URL { ModelRegistry.localPath(for: .transformer(.klein4B_8bit)) }

    private let weightsName = "diffusion_pytorch_model.safetensors"
    private var partURL: URL { int8Dir.appendingPathComponent("diffusion_pytorch_model.safetensors.part") }
    private var validatorURL: URL { int8Dir.appendingPathComponent("diffusion_pytorch_model.safetensors.part.validator") }

    /// A retained prefix of the transformer weights plus the sidecar naming `etag`.
    private func retainPrefix(_ prefix: Data, etag: String?) throws {
        try FileManager.default.createDirectory(at: int8Dir, withIntermediateDirectories: true)
        try prefix.write(to: partURL)
        if let etag {
            try JSONSerialization.data(withJSONObject: ["etag": etag]).write(to: validatorURL)
        }
    }

    private func weightRequests() -> [URLRequest] {
        StubHub.requests.filter { $0.url?.path.hasSuffix(weightsName) == true }
    }

    private func topLevelNames(_ dir: URL) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
    }

    // MARK: - Layout

    func testRootComponentDownloadsOnlyItsOwnTopLevelFiles() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)

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
        for name in int8Paths { try int8Files[name]!.write(to: dir.appendingPathComponent(name)) }

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

        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        _ = try await downloader().download(.transformer(.klein4B_8bit))

        XCTAssertEqual(topLevelNames(dir).subtracting([".mobile-diffuser-download-manifest.json"]),
                       ["config.json", "diffusion_pytorch_model.safetensors", "quanto_qmap.json"])
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("diffusion_pytorch_model.safetensors")),
                       transformerBody)
        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.transformer(.klein4B_8bit)))
    }

    // MARK: - Interruption and resume

    func testInterruptedFirstDownloadIsNotReportedInstalled() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        // The weights arrive; the last file of the component fails.
        StubHub.overrides["/\(int8Repo)/resolve/\(rev)/quanto_qmap.json"] = { _ in
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
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        try retainPrefix(transformerBody.prefix(1000), etag: StubHub.etag(transformerBody))

        _ = try await downloader().download(.transformer(.klein4B_8bit))

        let ranged = weightRequests().first
        XCTAssertEqual(ranged?.value(forHTTPHeaderField: "Range"), "bytes=1000-")
        XCTAssertEqual(ranged?.value(forHTTPHeaderField: "If-Range"), StubHub.etag(transformerBody))
        XCTAssertEqual(try Data(contentsOf: int8Dir.appendingPathComponent(weightsName)), transformerBody)
        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: validatorURL.path),
                       "the validator sidecar must go with the committed .part")
    }

    func testMisplacedPartialResponseIsRejectedBeforeAppend() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        try retainPrefix(transformerBody.prefix(1000), etag: StubHub.etag(transformerBody))
        let body = transformerBody
        StubHub.overrides["/\(int8Repo)/resolve/\(rev)/\(weightsName)"] = { _ in
            .init(status: 206, headers: ["Content-Range": "bytes 0-\(body.count - 1)/\(body.count)"], body: body)
        }

        do {
            _ = try await downloader().download(.transformer(.klein4B_8bit))
            XCTFail("a 206 that does not continue at the retained byte count must fail")
        } catch {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: int8Dir.appendingPathComponent(weightsName).path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: validatorURL.path))
    }

    // MARK: - Resume validator

    func testFreshDownloadSendsNoRangeAndLeavesNoSidecar() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)

        _ = try await downloader().download(.transformer(.klein4B_8bit))

        XCTAssertNil(weightRequests().first?.value(forHTTPHeaderField: "Range"))
        XCTAssertNil(weightRequests().first?.value(forHTTPHeaderField: "If-Range"))
        XCTAssertFalse(topLevelNames(int8Dir).contains { $0.hasSuffix(".part") || $0.hasSuffix(".validator") })
    }

    func testPartialWithoutValidatorRestartsFromZero() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        // Bytes that are NOT a prefix of the file: appending to them would fail verification.
        try retainPrefix(Data(repeating: 0xEE, count: 1000), etag: nil)

        _ = try await downloader().download(.transformer(.klein4B_8bit))

        XCTAssertNil(weightRequests().first?.value(forHTTPHeaderField: "Range"),
                     "a .part without a recorded validator cannot be resumed safely")
        XCTAssertEqual(try Data(contentsOf: int8Dir.appendingPathComponent(weightsName)), transformerBody)
    }

    func testChangedETagRestartsFromZero() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        // The retained prefix came from an older upload of the same path.
        try retainPrefix(Data(repeating: 0xEE, count: 1000), etag: "\"old-upload\"")

        _ = try await downloader().download(.transformer(.klein4B_8bit))

        let request = weightRequests().first
        XCTAssertEqual(request?.value(forHTTPHeaderField: "Range"), "bytes=1000-")
        XCTAssertEqual(request?.value(forHTTPHeaderField: "If-Range"), "\"old-upload\"")
        // The server answered 200; splicing the old prefix would have produced 5096 bytes.
        XCTAssertEqual(try Data(contentsOf: int8Dir.appendingPathComponent(weightsName)), transformerBody)
        XCTAssertFalse(FileManager.default.fileExists(atPath: validatorURL.path))
    }

    func testPartialResponseForAFileOfAnotherSizeIsRejected() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        let etag = StubHub.etag(transformerBody)
        try retainPrefix(transformerBody.prefix(1000), etag: etag)
        let body = transformerBody
        StubHub.overrides["/\(int8Repo)/resolve/\(rev)/\(weightsName)"] = { _ in
            .init(status: 206, headers: ["Content-Range": "bytes 1000-\(body.count - 1)/9999", "ETag": etag],
                  body: body.subdata(in: 1000..<body.count))
        }

        do {
            _ = try await downloader().download(.transformer(.klein4B_8bit))
            XCTFail("a 206 whose total differs from the listed size must fail")
        } catch {
            XCTAssertTrue("\(error.localizedDescription)".contains("9999"), "\(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: int8Dir.appendingPathComponent(weightsName).path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: validatorURL.path))
    }

    func testPartialResponseWithADifferentETagIsRejected() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        try retainPrefix(transformerBody.prefix(1000), etag: StubHub.etag(transformerBody))
        let body = transformerBody
        // A server that ignores If-Range and serves a range of a different upload.
        StubHub.overrides["/\(int8Repo)/resolve/\(rev)/\(weightsName)"] = { _ in
            .init(status: 206, headers: ["Content-Range": "bytes 1000-\(body.count - 1)/\(body.count)",
                                         "ETag": "\"another-upload\""],
                  body: body.subdata(in: 1000..<body.count))
        }

        do {
            _ = try await downloader().download(.transformer(.klein4B_8bit))
            XCTFail("a 206 naming a different ETag must fail")
        } catch {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: validatorURL.path))
    }

    func testRedirectToCDNKeepsRangeAndIfRange() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        let body = transformerBody
        let cdnETag = "\"cdn-object\""
        try retainPrefix(body.prefix(1000), etag: cdnETag)
        StubHub.overrides["/\(int8Repo)/resolve/\(rev)/\(weightsName)"] = { _ in
            .init(status: 302, headers: ["ETag": "\"hub-redirect\""], body: Data(),
                  redirect: URL(string: "https://cdn.stub.invalid/blob/weights")!)
        }
        StubHub.overrides["/blob/weights"] = { request in
            StubHub.serve(body, to: request, etag: cdnETag)
        }

        _ = try await downloader().download(.transformer(.klein4B_8bit))

        let cdn = StubHub.requests.first { $0.url?.host == "cdn.stub.invalid" }
        XCTAssertEqual(cdn?.value(forHTTPHeaderField: "Range"), "bytes=1000-")
        XCTAssertEqual(cdn?.value(forHTTPHeaderField: "If-Range"), cdnETag)
        XCTAssertEqual(try Data(contentsOf: int8Dir.appendingPathComponent(weightsName)), body)
    }

    // MARK: - Stale-file cleanup

    func testCleanupRemovesStrayWeightsAndStalePartialsButNotLinksOrDirectories() async throws {
        StubHub.publish(repo: int8Repo, revision: rev, files: int8Files)
        let dir = int8Dir
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(repeating: 0xB2, count: 30).write(to: dir.appendingPathComponent("model.safetensors"))
        try Data(repeating: 1, count: 10).write(to: dir.appendingPathComponent("old.safetensors.part"))
        try Data(#"{"etag":"\"x\""}"#.utf8).write(to: dir.appendingPathComponent("old.safetensors.part.validator"))
        try Data("keep".utf8).write(to: dir.appendingPathComponent("notes.txt"))
        // A directory whose name looks owned, and one holding weights: both stay.
        try fm.createDirectory(at: dir.appendingPathComponent("folder.safetensors"), withIntermediateDirectories: true)
        try fm.createDirectory(at: dir.appendingPathComponent("extra"), withIntermediateDirectories: true)
        try Data([1]).write(to: dir.appendingPathComponent("extra/inner.safetensors"))
        // A symlink to a file outside the directory: neither the link nor its target is removed.
        let outside = root.appendingPathComponent("outside.safetensors")
        try Data([2]).write(to: outside)
        try fm.createSymbolicLink(at: dir.appendingPathComponent("linked.safetensors"), withDestinationURL: outside)

        _ = try await downloader().download(.transformer(.klein4B_8bit))

        let names = topLevelNames(dir)
        XCTAssertFalse(names.contains("model.safetensors"))
        XCTAssertFalse(names.contains("old.safetensors.part"))
        XCTAssertFalse(names.contains("old.safetensors.part.validator"))
        XCTAssertTrue(names.contains("notes.txt"))
        XCTAssertTrue(names.contains("folder.safetensors"))
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("extra/inner.safetensors").path))
        XCTAssertNotNil(try? fm.destinationOfSymbolicLink(atPath: dir.appendingPathComponent("linked.safetensors").path))
        XCTAssertTrue(fm.fileExists(atPath: outside.path))
    }

    func testCleanupSkipsADirectoryThatResolvesOutsideTheModelsRoot() throws {
        let fm = FileManager.default
        let elsewhere = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-elsewhere-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: elsewhere, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: elsewhere) }
        let stray = elsewhere.appendingPathComponent("stray.safetensors")
        try Data([3]).write(to: stray)
        try fm.createDirectory(at: int8Dir.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fm.createSymbolicLink(at: int8Dir, withDestinationURL: elsewhere)

        XCTAssertNil(Flux2ModelDownloader.managedDirectory(for: .transformer(.klein4B_8bit)))
        Flux2ModelDownloader.removeStaleFiles(for: .transformer(.klein4B_8bit), keeping: [], previouslyRecorded: [])
        XCTAssertTrue(fm.fileExists(atPath: stray.path))
    }
}
