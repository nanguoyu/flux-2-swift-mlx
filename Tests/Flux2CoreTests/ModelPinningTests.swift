// ModelPinningTests.swift - Pinned revisions and file tables for the Flux components (no network)

import CryptoKit
import XCTest
import FluxTextEncoders
@testable import Flux2Core

final class ModelPinningTests: XCTestCase {
    private var root: URL!
    private let rev = "fedcba9876543210fedcba9876543210fedcba98"
    private let repo = ModelRegistry.VAEVariant.standard.huggingFaceRepo
    private let manifestName = ".mobile-diffuser-download-manifest.json"

    // A Klein-shaped repository: the standard VAE lives under vae/.
    private let weights = Data((0..<4096).map { UInt8($0 % 241) })
    private var repoFiles: [String: Data] {
        [
            "model_index.json": Data(#"{"root":1}"#.utf8),
            "flux-2-klein-4b.safetensors": Data(repeating: 0xA1, count: 2048),
            "vae/config.json": Data(#"{"component":"vae"}"#.utf8),
            "vae/diffusion_pytorch_model.safetensors": weights,
        ]
    }
    private let vaePaths = ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
    private var vaeDir: URL { ModelRegistry.localPath(for: .vae(.standard)) }

    override func setUpWithError() throws {
        StubHub.reset()
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-pinning-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ModelRegistry.customModelsDirectory = root
        let pin = StubHub.pin(revision: rev, files: repoFiles, paths: vaePaths)
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

    private func names(_ dir: URL) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
    }

    // MARK: - URLs

    func testListingAndResolveURLsCarryThePinnedRevision() async throws {
        StubHub.publish(repo: repo, revision: rev, files: repoFiles)

        let dir = try await downloader().download(.vae(.standard))

        let paths = StubHub.recordedPaths
        XCTAssertEqual(paths.filter { $0.hasPrefix("/api/models/") }, ["/api/models/\(repo)/tree/\(rev)/vae"])
        XCTAssertEqual(Set(paths.filter { $0.contains("/resolve/") }),
                       Set(vaePaths.map { "/\(repo)/resolve/\(rev)/\($0)" }))
        XCTAssertFalse(paths.contains { $0.contains("/main") }, "a pinned component never reads main")
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("diffusion_pytorch_model.safetensors")), weights)

        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: dir.appendingPathComponent(manifestName))) as? [String: Any]
        XCTAssertEqual(manifest?["version"] as? Int, 2)
        XCTAssertEqual(manifest?["revision"] as? String, rev)
        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    // MARK: - Integrity

    func testTamperedFileOfTheRightSizeIsRejectedAndNotInstalled() async throws {
        // Upstream now serves different bytes of the same size (and lists their hash): the pin wins.
        var served = repoFiles
        var tampered = weights
        tampered[100] ^= 0xFF
        served["vae/diffusion_pytorch_model.safetensors"] = tampered
        StubHub.publish(repo: repo, revision: rev, files: served)

        do {
            _ = try await downloader().download(.vae(.standard))
            XCTFail("a file whose hash differs from the pin must be rejected")
        } catch {
            XCTAssertTrue("\(error.localizedDescription)".contains("diffusion_pytorch_model.safetensors"), "\(error)")
        }
        let weightsURL = vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        XCTAssertFalse(FileManager.default.fileExists(atPath: weightsURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: weightsURL.appendingPathExtension("part").path))
        XCTAssertFalse(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    func testTamperedSmallFileFailsTheInstalledCheck() async throws {
        StubHub.publish(repo: repo, revision: rev, files: repoFiles)
        let dir = try await downloader().download(.vae(.standard))
        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.vae(.standard)))

        // Same size, different content: small files are hashed on every lookup.
        try Data(#"{"component":"VAE"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
        XCTAssertFalse(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    func testUnexpectedExtraUpstreamFilesAreIgnored() async throws {
        var served = repoFiles
        served["vae/extra.safetensors"] = Data(repeating: 9, count: 64)
        served["vae/new_config.json"] = Data("{}".utf8)
        StubHub.publish(repo: repo, revision: rev, files: served)

        let dir = try await downloader().download(.vae(.standard))

        XCTAssertEqual(names(dir).subtracting([manifestName]), ["config.json", "diffusion_pytorch_model.safetensors"])
        XCTAssertFalse(StubHub.recordedPaths.contains { $0.hasSuffix("extra.safetensors") || $0.hasSuffix("new_config.json") })
    }

    func testListingThatLacksAPinnedFileFailsBeforeAnyFetch() async throws {
        var served = repoFiles
        served["vae/config.json"] = nil
        StubHub.publish(repo: repo, revision: rev, files: served)

        do {
            _ = try await downloader().download(.vae(.standard))
            XCTFail("a listing without a pinned file must fail")
        } catch {
            let message = error.localizedDescription
            XCTAssertTrue(message.contains("vae/config.json") && message.contains(rev), message)
        }
        XCTAssertFalse(StubHub.recordedPaths.contains { $0.contains("/resolve/") })
        XCTAssertFalse(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    // MARK: - Existing installs

    func testOldManifestInstallMatchingThePinVerifiesWithoutNetwork() async throws {
        // What the unpinned downloader left: the files plus a version 1 manifest without a revision,
        // with an LFS hash for the weights only.
        let dir = vaeDir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for path in vaePaths {
            try repoFiles[path]!.write(to: dir.appendingPathComponent(URL(fileURLWithPath: path).lastPathComponent))
        }
        let weightsSHA = SHA256.hash(data: weights).map { String(format: "%02x", $0) }.joined()
        let manifest: [String: Any] = ["version": 1, "files": [
            ["filename": "config.json", "size": repoFiles["vae/config.json"]!.count],
            ["filename": "diffusion_pytorch_model.safetensors", "size": weights.count, "sha256": weightsSHA],
        ]]
        try JSONSerialization.data(withJSONObject: manifest).write(to: dir.appendingPathComponent(manifestName))

        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
        _ = try await downloader().download(.vae(.standard))
        XCTAssertTrue(StubHub.recordedPaths.isEmpty, "a pinned install must not touch the network")
    }

    func testInstallWhoseWeightsDifferFromThePinIsNotInstalled() throws {
        let dir = vaeDir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try repoFiles["vae/config.json"]!.write(to: dir.appendingPathComponent("config.json"))
        try Data(repeating: 1, count: 10).write(to: dir.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        XCTAssertFalse(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    func testStrayWeightsBesideThePinnedFilesAreNotInstalledAndAreRemovedWithoutRefetch() async throws {
        // The old recursive layout could leave another component's shard beside correct files.
        let dir = vaeDir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for path in vaePaths {
            try repoFiles[path]!.write(to: dir.appendingPathComponent(URL(fileURLWithPath: path).lastPathComponent))
        }
        try Data(repeating: 0xB2, count: 300).write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        XCTAssertFalse(Flux2ModelDownloader.isDownloaded(.vae(.standard)))

        StubHub.publish(repo: repo, revision: rev, files: repoFiles)
        _ = try await downloader().download(.vae(.standard))

        XCTAssertFalse(StubHub.recordedPaths.contains { $0.contains("/resolve/") },
                       "files already matching the pin are kept, not fetched again")
        XCTAssertEqual(names(dir).subtracting([manifestName]), ["config.json", "diffusion_pytorch_model.safetensors"])
        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    // MARK: - Unpinned variants

    func testUnpinnedComponentNeedsTheOptIn() async throws {
        Flux2ModelDownloader.pinOverrideForTesting = { _ in nil }
        StubHub.publish(repo: repo, revision: "main", files: repoFiles)

        do {
            _ = try await downloader().download(.vae(.standard))
            XCTFail("an unpinned component must not download without the opt-in")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("allowsUnpinnedRevisions"), "\(error)")
        }
        XCTAssertTrue(StubHub.recordedPaths.isEmpty)

        HubDownloadPolicy.allowsUnpinnedRevisions = true
        _ = try await downloader().download(.vae(.standard))
        XCTAssertTrue(StubHub.recordedPaths.contains("/api/models/\(repo)/tree/main/vae"))
        XCTAssertTrue(Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    // MARK: - The shipped tables

    func testShippedComponentsArePinnedToFullCommits() {
        Flux2ModelDownloader.pinOverrideForTesting = nil
        let expected: [(ModelRegistry.ModelComponent, String)] = [
            (.transformer(.klein4B_bf16), "e7b7dc27f91deacad38e78976d1f2b499d76a294"),
            (.transformer(.klein4B_8bit), "f38839c94b597211dea83b1326f1bb4ce4142524"),
            (.transformer(.klein4B_4bit), "860e87183ceb29e39627c0612ebd66d8ea66e68c"),
            (.vae(.standard), "e7b7dc27f91deacad38e78976d1f2b499d76a294"),
            (.vae(.smallDecoder), "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a"),
        ]
        let hex = CharacterSet(charactersIn: "0123456789abcdef")
        for (component, revision) in expected {
            guard let pin = component.pinnedSnapshot else { XCTFail("\(component) is unpinned"); continue }
            XCTAssertEqual(pin.revision, revision, "\(component)")
            XCTAssertFalse(pin.files.isEmpty)
            XCTAssertEqual(Set(pin.files.map(\.filename)).count, pin.files.count, "names collide when flattened")
            for file in pin.files {
                XCTAssertEqual(file.sha256.count, 64, file.path)
                XCTAssertTrue(file.sha256.unicodeScalars.allSatisfy(hex.contains), file.path)
                XCTAssertGreaterThan(file.size, 0, file.path)
            }
            // The table is exactly what the downloader's own filter keeps from those paths.
            let subfolder: String? = {
                switch component {
                case .transformer(let v): return v.huggingFaceSubfolder
                case .vae(let v): return v.huggingFaceSubfolder
                case .textEncoder: return nil
                }
            }()
            let asListed = pin.files.map { Flux2ModelDownloader.HubFile(path: $0.path, size: $0.size, sha256: $0.sha256) }
            XCTAssertEqual(Flux2ModelDownloader.componentFiles(from: asListed, subfolder: subfolder).map(\.path),
                           pin.files.map(\.path), "\(component)")
        }
        XCTAssertNil(ModelRegistry.ModelComponent.transformer(.klein9B_bf16).pinnedSnapshot)
    }
}
