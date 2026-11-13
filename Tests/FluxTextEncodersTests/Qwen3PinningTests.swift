// Qwen3PinningTests.swift - Pinned revisions and file tables for the Qwen3 encoders (no network)

import CryptoKit
import XCTest
@testable import FluxTextEncoders

final class Qwen3PinningTests: XCTestCase {
    // A variant that is not in a developer's ~/.cache/huggingface, so only our temp root counts.
    private let variant = Qwen3Variant.qwen3_8B_4bit
    private let rev = "13579bdf13579bdf13579bdf13579bdf13579bdf"
    private let weights = Data((0..<5000).map { UInt8($0 % 239) })
    private let pinnedNames = ["config.json", "model.safetensors", "tokenizer.json"]
    private let manifestName = ".mobile-diffuser-download-manifest.json"
    private var root: URL!

    private var publishedFiles: [String: Data] {
        [
            "config.json": Data(#"{"model_type":"qwen3"}"#.utf8),
            "model.safetensors": weights,
            "tokenizer.json": Data(#"{"t":1}"#.utf8),
        ]
    }

    override func setUpWithError() throws {
        QwenStubHub.reset()
        QwenStubHub.repo = variant.repoId
        QwenStubHub.revision = rev
        QwenStubHub.files = publishedFiles
        let pin = QwenStubHub.pin(pinnedNames)
        TextEncoderModelDownloader.qwen3PinOverrideForTesting = { _ in pin }
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-pinning-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        TextEncoderModelDownloader.customModelsDirectory = root
    }

    override func tearDownWithError() throws {
        TextEncoderModelDownloader.qwen3PinOverrideForTesting = nil
        HubDownloadPolicy.allowsUnpinnedRevisions = false
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

    private var paths: [String] { QwenStubHub.requests.compactMap { $0.url?.path } }

    func testListingAndResolveURLsCarryThePinnedRevision() async throws {
        _ = try await downloader().downloadQwen3(variant: variant)

        XCTAssertEqual(paths.filter { $0.hasPrefix("/api/models/") }, ["/api/models/\(variant.repoId)/tree/\(rev)"])
        XCTAssertEqual(Set(paths.filter { $0.contains("/resolve/") }),
                       Set(pinnedNames.map { "/\(variant.repoId)/resolve/\(rev)/\($0)" }))
        XCTAssertFalse(paths.contains { $0.contains("/main") })
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: repoDir.appendingPathComponent(manifestName))) as? [String: Any]
        XCTAssertEqual(manifest?["revision"] as? String, rev)
        XCTAssertTrue(TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: variant))
    }

    func testTamperedFileOfTheRightSizeIsRejectedAndNotInstalled() async throws {
        var tampered = weights
        tampered[10] ^= 0xFF
        QwenStubHub.files["model.safetensors"] = tampered

        do {
            _ = try await downloader().downloadQwen3(variant: variant)
            XCTFail("a file whose hash differs from the pin must be rejected")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("model.safetensors"), "\(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: repoDir.appendingPathComponent("model.safetensors").path))
        XCTAssertFalse(TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: variant))
    }

    func testUnexpectedExtraUpstreamFilesAreIgnored() async throws {
        QwenStubHub.files["extra.safetensors"] = Data(repeating: 3, count: 64)
        QwenStubHub.files["generation_config.json"] = Data("{}".utf8)

        _ = try await downloader().downloadQwen3(variant: variant)

        let names = Set((try? FileManager.default.contentsOfDirectory(atPath: repoDir.path)) ?? [])
        XCTAssertEqual(names.subtracting([manifestName]), Set(pinnedNames))
        XCTAssertFalse(paths.contains { $0.hasSuffix("extra.safetensors") || $0.hasSuffix("generation_config.json") })
    }

    func testListingThatLacksAPinnedFileFailsBeforeAnyFetch() async throws {
        QwenStubHub.files["tokenizer.json"] = nil

        do {
            _ = try await downloader().downloadQwen3(variant: variant)
            XCTFail("a listing without a pinned file must fail")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("tokenizer.json"), "\(error)")
        }
        XCTAssertFalse(paths.contains { $0.contains("/resolve/") })
    }

    func testOldManifestInstallMatchingThePinVerifiesWithoutNetwork() async throws {
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        for name in pinnedNames { try publishedFiles[name]!.write(to: repoDir.appendingPathComponent(name)) }
        // A version 1 manifest from the unpinned listing: sizes, and a hash for the weights only.
        let sha = SHA256.hash(data: weights).map { String(format: "%02x", $0) }.joined()
        let manifest: [String: Any] = ["version": 1, "files": [
            ["filename": "config.json", "size": publishedFiles["config.json"]!.count],
            ["filename": "model.safetensors", "size": weights.count, "sha256": sha],
            ["filename": "tokenizer.json", "size": publishedFiles["tokenizer.json"]!.count],
        ]]
        try JSONSerialization.data(withJSONObject: manifest).write(to: repoDir.appendingPathComponent(manifestName))

        XCTAssertTrue(TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: variant))
        _ = try await downloader().downloadQwen3(variant: variant)
        XCTAssertTrue(QwenStubHub.requests.isEmpty, "a pinned install must not touch the network")
    }

    func testUnpinnedVariantNeedsTheOptIn() async throws {
        TextEncoderModelDownloader.qwen3PinOverrideForTesting = { _ in nil }
        QwenStubHub.revision = "main"

        do {
            _ = try await downloader().downloadQwen3(variant: variant)
            XCTFail("an unpinned variant must not download without the opt-in")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("allowsUnpinnedRevisions"), "\(error)")
        }
        XCTAssertTrue(QwenStubHub.requests.isEmpty)

        HubDownloadPolicy.allowsUnpinnedRevisions = true
        _ = try await downloader().downloadQwen3(variant: variant)
        XCTAssertTrue(paths.contains("/api/models/\(variant.repoId)/tree/main"))
    }

    func testShippedVariantsArePinnedToFullCommits() {
        TextEncoderModelDownloader.qwen3PinOverrideForTesting = nil
        let expected: [(Qwen3Variant, String)] = [
            (.qwen3_4B_4bit, "480ec45ecf06ae3c061e4ef11345d05c7fdb02f1"),
            (.qwen3_4B_8bit, "65ade2e73183ae9123edd7e5debabe260b21422a"),
        ]
        for (variant, revision) in expected {
            guard let pin = variant.pinnedSnapshot else { XCTFail("\(variant) is unpinned"); continue }
            XCTAssertEqual(pin.revision, revision)
            XCTAssertEqual(Set(pin.files.map(\.path)), [
                "added_tokens.json", "config.json", "model.safetensors", "model.safetensors.index.json",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
            ])
            for file in pin.files {
                XCTAssertEqual(file.sha256.count, 64, file.path)
                XCTAssertFalse(file.path.contains("/"), file.path)
            }
        }
        XCTAssertNil(Qwen3Variant.qwen3_8B_8bit.pinnedSnapshot)
        // A model info pointing at another repository never borrows the variant's pin.
        XCTAssertNil(TextEncoderModelDownloader.qwen3Pin(for: .qwen3_4B_4bit, repoId: "someone/else"))
    }
}
