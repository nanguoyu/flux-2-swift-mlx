/**
 * TextEncoderModelDownloader.swift
 * Downloads Mistral and Qwen3 models from HuggingFace Hub for text encoding
 */

import Foundation
import Hub
import CryptoKit

/// Progress callback for download updates
public typealias TextEncoderDownloadProgressCallback = @Sendable (Double, String) -> Void

/// Model downloader with HuggingFace Hub integration
public class TextEncoderModelDownloader {
    private struct HubFile: Sendable {
        var path: String
        var size: Int64?
        var sha256: String?
    }

    private struct DownloadManifest: Codable {
        struct File: Codable {
            var filename: String
            var size: Int64?
            var sha256: String?
        }
        var version: Int = 1
        var files: [File]
    }

    private static let manifestFilename = ".mobile-diffuser-download-manifest.json"

    /// HuggingFace token for private/gated models
    private var hfToken: String?

    private let session: URLSession

    /// Custom override for model storage directory.
    /// Set this before any download/check call to redirect model storage.
    nonisolated(unsafe) public static var customModelsDirectory: URL?

    /// Hub API instance — recreated if custom directory is set
    nonisolated(unsafe) private static var hubApi: HubApi = makeHubApi()

    private static func makeHubApi() -> HubApi {
        setenv("CI_DISABLE_NETWORK_MONITOR", "1", 1)
        let base = customModelsDirectory?.deletingLastPathComponent()
            ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        return HubApi(downloadBase: base)
    }

    /// Call after setting customModelsDirectory to update the HubApi
    public static func reconfigureHubApi() {
        hubApi = makeHubApi()
    }

    /// Directory where HubApi downloads models.
    /// Uses customModelsDirectory if set, otherwise falls back to ~/Library/Caches/models
    private static var hubDownloadDirectory: URL {
        if let custom = customModelsDirectory {
            return custom
        }
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("models")
    }

    /// Cross-platform base for the default/legacy model lookups. macOS has a real user home
    /// (`~/.mistral`, `~/.cache/huggingface`); iOS has none (`homeDirectoryForCurrentUser` is
    /// unavailable there), so fall back to the app's caches directory — the legacy
    /// `~/.cache/huggingface` paths simply won't exist on iOS, which is correct (models download
    /// fresh into the caches dir).
    static var legacyHomeDirectory: URL {
        #if os(iOS)
        return FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        #else
        return FileManager.default.homeDirectoryForCurrentUser
        #endif
    }

    /// Legacy models directory (Mistral).
    /// Uses customModelsDirectory if set, otherwise falls back to ~/.mistral/models
    public static var modelsDirectory: URL {
        if let custom = customModelsDirectory {
            return custom
        }
        let homeDir = legacyHomeDirectory
        return homeDir.appendingPathComponent(".mistral").appendingPathComponent("models")
    }

    public init(hfToken: String? = nil) {
        self.hfToken = hfToken
        if let token = hfToken {
            setenv("HF_TOKEN", token, 1)
        }
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 3600
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: config)
    }

    /// Check if a model is already downloaded
    public static func isModelDownloaded(_ model: ModelInfo) -> Bool {
        return findModelPath(for: model) != nil
    }

    /// Get the HuggingFace Hub cache path for a model
    public static func hubCachePath(for model: ModelInfo) -> URL? {
        // Check Hub download directory: {hubDownloadDirectory}/{org}/{repo}
        var newPath = hubDownloadDirectory
        for component in model.repoId.split(separator: "/") {
            newPath = newPath.appendingPathComponent(String(component))
        }

        if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
            return newPath
        }

        // Check legacy location: ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/...
        let homeDir = legacyHomeDirectory
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        let configPath = modelPath.appendingPathComponent("config.json")

        if FileManager.default.fileExists(atPath: configPath.path) {
            return modelPath
        }

        return nil
    }

    /// Find a model path (checks Hub cache first, then local directory)
    public static func findModelPath(for model: ModelInfo) -> URL? {
        // Check Hub cache first
        if let hubPath = hubCachePath(for: model) {
            let verification = verifyShardedModel(at: hubPath)
            if verification.complete {
                return hubPath
            }
        }

        // Check local models directory
        let localDir = modelsDirectory.appendingPathComponent(model.repoId.replacingOccurrences(of: "/", with: "--"))
        if FileManager.default.fileExists(atPath: localDir.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: localDir)
            if verification.complete {
                return localDir
            }
        }

        return nil
    }

    /// Verify that a sharded model has all required safetensors files
    /// Note: Does NOT trust index.json as some HF repos have mismatched index files
    /// Instead, detects safetensors files and verifies the series is complete
    public static func verifyShardedModel(at path: URL) -> (complete: Bool, missing: [String]) {
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: path.path)) ?? []
        let safetensorsFiles = contents.filter { $0.hasSuffix(".safetensors") }

        // Single file model
        if safetensorsFiles.contains("model.safetensors") {
            return (true, [])
        }

        // No safetensors files at all
        guard !safetensorsFiles.isEmpty else {
            return (false, ["No safetensors files found"])
        }

        // Parse sharded file pattern: model-XXXXX-of-YYYYY.safetensors
        // Example: model-00001-of-00003.safetensors
        var totalShards: Int?
        var foundIndices: Set<Int> = []

        for file in safetensorsFiles {
            // Parse filename like "model-00001-of-00003.safetensors"
            let name = file.replacingOccurrences(of: ".safetensors", with: "")
            let parts = name.split(separator: "-")
            // Expected: ["model", "00001", "of", "00003"]
            guard parts.count == 4,
                  parts[0] == "model",
                  parts[2] == "of",
                  let index = Int(parts[1]),
                  let total = Int(parts[3]) else {
                continue
            }

            if totalShards == nil {
                totalShards = total
            } else if totalShards != total {
                // Inconsistent totals - mixed files
                return (false, ["Inconsistent shard totals: \(totalShards!) vs \(total)"])
            }

            foundIndices.insert(index)
        }

        // If we found sharded files, verify all parts are present
        if let total = totalShards {
            let expectedIndices = Set(1...total)
            let missing = expectedIndices.subtracting(foundIndices)

            if missing.isEmpty {
                return (true, [])
            } else {
                let missingFiles = missing.sorted().map { "model-\(String(format: "%05d", $0))-of-\(String(format: "%05d", total)).safetensors" }
                return (false, missingFiles)
            }
        }

        // Has some safetensors files but not in standard sharded format
        // Consider it complete if there are any safetensors files
        return (true, [])
    }

    /// Download a model using Hub API
    public func download(
        _ model: ModelInfo,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = Self.findModelPath(for: model) {
            let verification = Self.verifyShardedModel(at: existingPath)
            if verification.complete {
                // Also ensure tekken.json exists
                await ensureTekkenJson(at: existingPath, progress: progress)
                progress?(1.0, "Model already downloaded")
                return existingPath
            } else {
                print("Warning: Incomplete download detected. Missing files: \(verification.missing)")
                print("Re-downloading...")
            }
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        let modelUrl = try await Self.hubApi.snapshot(
            from: model.repoId,
            matching: ["*.json", "*.safetensors"]
        ) { downloadProgress in
            let completed = downloadProgress.completedUnitCount
            let total = downloadProgress.totalUnitCount
            let fraction = downloadProgress.fractionCompleted

            let message = "Downloading file \(completed)/\(total)"
            progress?(fraction, message)
        }

        let verification = Self.verifyShardedModel(at: modelUrl)
        if !verification.complete {
            print("\nWarning: Download may be incomplete. Missing files: \(verification.missing)")
        }

        // Download tekken.json from original Mistral repo if not present
        await ensureTekkenJson(at: modelUrl, progress: progress)

        progress?(1.0, "Download complete!")
        print("\nDownload complete: \(modelUrl.path)")

        return modelUrl
    }

    /// Ensure tekken.json exists in the model directory
    /// Downloads from original Mistral repo if not present
    private func ensureTekkenJson(at modelPath: URL, progress: TextEncoderDownloadProgressCallback? = nil) async {
        let tekkenPath = modelPath.appendingPathComponent("tekken.json")

        // Check if already exists
        if FileManager.default.fileExists(atPath: tekkenPath.path) {
            // Verify it's not a Git LFS pointer
            if let data = try? Data(contentsOf: tekkenPath),
               data.count > 1000 {  // Real file is ~19MB, pointer is < 200 bytes
                return
            }
        }

        progress?(0.9, "Downloading tekken.json tokenizer...")
        print("Downloading tekken.json from Mistral AI repository...")

        let tekkenUrl = URL(string: "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/resolve/main/tekken.json")!

        do {
            let (data, response) = try await URLSession.shared.data(from: tekkenUrl)

            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode == 200,
               data.count > 1000 {  // Sanity check - real file is ~19MB
                try data.write(to: tekkenPath)
                print("tekken.json downloaded successfully (\(data.count / 1_000_000)MB)")
            } else {
                print("Warning: Failed to download tekken.json - response invalid")
            }
        } catch {
            print("Warning: Could not download tekken.json: \(error.localizedDescription)")
        }
    }

    /// Download a model by variant
    public func download(
        variant: ModelVariant,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        guard let model = await TextEncoderModelRegistry.shared.model(withVariant: variant) else {
            throw TextEncoderModelDownloaderError.modelNotFound
        }
        return try await download(model, progress: progress)
    }

    // MARK: - Qwen3 Downloads

    /// Download a Qwen3 model for Klein embeddings
    public func downloadQwen3(
        _ model: Qwen3ModelInfo,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = Self.findQwen3ModelPath(for: model) {
            let verification = Self.verifyShardedModel(at: existingPath)
            if verification.complete {
                progress?(1.0, "Qwen3 model already downloaded")
                return existingPath
            } else {
                print("Warning: Incomplete Qwen3 download detected. Missing files: \(verification.missing)")
                print("Re-downloading...")
            }
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        let modelUrl = try await downloadResumableSnapshot(
            repoId: model.repoId,
            matching: { path in path.hasSuffix(".json") || path.hasSuffix(".safetensors") || path == "tokenizer.model" },
            progress: progress
        )

        let verification = Self.verifyShardedModel(at: modelUrl)
        if !verification.complete {
            print("\nWarning: Qwen3 download may be incomplete. Missing files: \(verification.missing)")
        }

        progress?(1.0, "Download complete!")
        print("\nQwen3 download complete: \(modelUrl.path)")

        return modelUrl
    }

    private func downloadResumableSnapshot(
        repoId: String,
        matching: @escaping (String) -> Bool,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        let allFiles = try await fetchHubFiles(repoId: repoId)
        let files = allFiles.filter { matching($0.path) }
        guard !files.isEmpty else { throw TextEncoderModelDownloaderError.downloadFailed("No model files found in \(repoId)") }

        var destDir = Self.hubDownloadDirectory
        for part in repoId.split(separator: "/") { destDir = destDir.appendingPathComponent(String(part)) }
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let totalBytes = max(1, files.reduce(Int64(0)) { $0 + ($1.size ?? 0) })
        var completedBytes: Int64 = 0
        for file in files {
            let filename = URL(fileURLWithPath: file.path).lastPathComponent
            let destination = destDir.appendingPathComponent(filename)
            if Self.fileMatches(destination, expectedSize: file.size, expectedSHA256: file.sha256) {
                completedBytes += file.size ?? Self.fileSize(destination)
                progress?(min(1, Double(completedBytes) / Double(totalBytes)), "Verified \(filename)")
                continue
            }
            let baseBytes = completedBytes
            try await downloadFile(repoId: repoId, filePath: file.path, to: destination,
                                   expectedSize: file.size, expectedSHA256: file.sha256) { bytes in
                progress?(min(1, Double(baseBytes + bytes) / Double(totalBytes)), "Downloading \(filename)")
            }
            completedBytes += Self.fileSize(destination)
        }
        try Self.writeManifest(files, at: destDir)
        return destDir
    }

    private func fetchHubFiles(repoId: String) async throws -> [HubFile] {
        let urlString = "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=1&expand=1"
        guard let url = URL(string: urlString) else {
            throw TextEncoderModelDownloaderError.downloadFailed("Invalid URL: \(urlString)")
        }
        var request = URLRequest(url: url)
        if let token = hfToken { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw TextEncoderModelDownloaderError.downloadFailed("Could not list files for \(repoId)")
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw TextEncoderModelDownloaderError.downloadFailed("Invalid file list for \(repoId)")
        }
        return json.compactMap { item in
            guard (item["type"] as? String) == "file", let path = item["path"] as? String else { return nil }
            let size = (item["size"] as? NSNumber)?.int64Value
            let lfs = item["lfs"] as? [String: Any]
            let sha256 = lfs?["oid"] as? String
            let lfsSize = (lfs?["size"] as? NSNumber)?.int64Value
            return HubFile(path: path, size: lfsSize ?? size, sha256: sha256)
        }
    }

    private func downloadFile(repoId: String, filePath: String, to destination: URL,
                              expectedSize: Int64?, expectedSHA256: String?,
                              progress: @escaping @Sendable (Int64) -> Void) async throws {
        if Self.fileMatches(destination, expectedSize: expectedSize, expectedSHA256: expectedSHA256) {
            progress(expectedSize ?? Self.fileSize(destination)); return
        }
        let urlString = "https://huggingface.co/\(repoId)/resolve/main/\(filePath)"
        guard let url = URL(string: urlString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? urlString) else {
            throw TextEncoderModelDownloaderError.downloadFailed("Invalid URL: \(urlString)")
        }
        var request = URLRequest(url: url)
        if let token = hfToken { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let partURL = destination.appendingPathExtension("part")
        var existingBytes = Self.fileSize(partURL)
        if existingBytes > 0 { request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range") }

        // Native chunked download to a temp file (URLSession.bytes' per-UInt8 AsyncSequence is far too
        // slow for multi-GB encoder shards). Resume is handled via the Range header + the .part file.
        let (tempURL, response) = try await session.download(for: request)
        defer { try? FileManager.default.removeItem(at: tempURL) }
        guard let http = response as? HTTPURLResponse else {
            throw TextEncoderModelDownloaderError.downloadFailed("Invalid response for \(filePath)")
        }
        if existingBytes > 0, http.statusCode != 206 {
            // Server ignored the Range and sent the full file — discard the partial to avoid corruption.
            try? FileManager.default.removeItem(at: partURL)
            existingBytes = 0
        }
        guard http.statusCode == 200 || http.statusCode == 206 else {
            throw TextEncoderModelDownloaderError.downloadFailed("HTTP \(http.statusCode) for \(filePath)")
        }
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if existingBytes == 0 {
            if FileManager.default.fileExists(atPath: partURL.path) { try FileManager.default.removeItem(at: partURL) }
            try FileManager.default.moveItem(at: tempURL, to: partURL)
        } else {
            let outHandle = try FileHandle(forWritingTo: partURL)
            let inHandle = try FileHandle(forReadingFrom: tempURL)
            defer { try? outHandle.close(); try? inHandle.close() }
            try outHandle.seekToEnd()
            while let chunk = try inHandle.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty {
                try outHandle.write(contentsOf: chunk)
            }
        }
        progress(Self.fileSize(partURL))
        guard Self.fileMatches(partURL, expectedSize: expectedSize, expectedSHA256: expectedSHA256) else {
            // Remove the corrupt partial so a retry re-downloads from scratch, not the same bad prefix.
            try? FileManager.default.removeItem(at: partURL)
            throw TextEncoderModelDownloaderError.downloadFailed("Hash or size verification failed for \(filePath)")
        }
        if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
        try FileManager.default.moveItem(at: partURL, to: destination)
    }

    private static func writeManifest(_ files: [HubFile], at directory: URL) throws {
        let manifest = DownloadManifest(files: files.map {
            DownloadManifest.File(filename: URL(fileURLWithPath: $0.path).lastPathComponent,
                                  size: $0.size, sha256: $0.sha256)
        })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: directory.appendingPathComponent(manifestFilename), options: .atomic)
    }

    private static func verifyManifestIfPresent(at directory: URL, verifyHashes: Bool = false) -> Bool {
        let url = directory.appendingPathComponent(manifestFilename)
        // No manifest (older app version, mflux, HF CLI cache) or an unreadable one is not evidence of
        // a bad download — trust the structural check rather than forcing a multi-GB re-download. Only a
        // readable manifest whose listed files fail size/hash means the download is actually incomplete.
        guard FileManager.default.fileExists(atPath: url.path) else { return true }
        guard let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(DownloadManifest.self, from: data) else { return true }
        for file in manifest.files {
            guard fileMatches(directory.appendingPathComponent(file.filename), expectedSize: file.size, expectedSHA256: file.sha256, verifyHash: verifyHashes) else {
                return false
            }
        }
        return true
    }

    private static func fileMatches(_ url: URL, expectedSize: Int64?, expectedSHA256: String?, verifyHash: Bool = true) -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        if let expectedSize, fileSize(url) != expectedSize { return false }
        if verifyHash, let expectedSHA256, !expectedSHA256.isEmpty {
            guard (try? sha256Hex(of: url)) == expectedSHA256.lowercased() else { return false }
        }
        return true
    }

    private static func fileSize(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
    }

    private static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 1024 * 1024) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Download a Qwen3 model by variant
    public func downloadQwen3(
        variant: Qwen3Variant,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        guard let model = await TextEncoderModelRegistry.shared.qwen3Model(withVariant: variant) else {
            throw TextEncoderModelDownloaderError.qwen3ModelNotFound
        }
        return try await downloadQwen3(model, progress: progress)
    }

    /// Find a Qwen3 model path (checks Hub cache)
    /// Verifies safetensors files are complete before returning path
    public static func findQwen3ModelPath(for model: Qwen3ModelInfo) -> URL? {
        // Check Hub download directory: {hubDownloadDirectory}/{org}/{repo}
        var newPath = hubDownloadDirectory
        for component in model.repoId.split(separator: "/") {
            newPath = newPath.appendingPathComponent(String(component))
        }

        if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
            // Verify safetensors files are complete
            let verification = verifyShardedModel(at: newPath)
            if verification.complete && verifyManifestIfPresent(at: newPath) {
                return newPath
            }
        }

        // Check legacy location
        let homeDir = legacyHomeDirectory
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        let configPath = modelPath.appendingPathComponent("config.json")

        if FileManager.default.fileExists(atPath: configPath.path) {
            // Verify safetensors files are complete
            let verification = verifyShardedModel(at: modelPath)
            if verification.complete && verifyManifestIfPresent(at: modelPath) {
                return modelPath
            }
        }

        return nil
    }

    /// Check if a Qwen3 model is already downloaded
    public static func isQwen3ModelDownloaded(_ model: Qwen3ModelInfo) -> Bool {
        return findQwen3ModelPath(for: model) != nil
    }

    /// Find a Qwen3 model path by variant
    /// Verifies safetensors files are complete before returning path
    public static func findQwen3ModelPath(for variant: Qwen3Variant) -> URL? {
        let repoId = variant.repoId

        // Check Hub download directory: {hubDownloadDirectory}/{org}/{repo}
        var newPath = hubDownloadDirectory
        for component in repoId.split(separator: "/") {
            newPath = newPath.appendingPathComponent(String(component))
        }

        if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
            // Verify safetensors files are complete
            let verification = verifyShardedModel(at: newPath)
            if verification.complete && verifyManifestIfPresent(at: newPath) {
                return newPath
            }
        }

        // Check legacy location
        let homeDir = legacyHomeDirectory
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        let configPath = modelPath.appendingPathComponent("config.json")

        if FileManager.default.fileExists(atPath: configPath.path) {
            // Verify safetensors files are complete
            let verification = verifyShardedModel(at: modelPath)
            if verification.complete && verifyManifestIfPresent(at: modelPath) {
                return modelPath
            }
        }

        return nil
    }

    /// Check if a Qwen3 model is already downloaded by variant
    public static func isQwen3ModelDownloaded(variant: Qwen3Variant) -> Bool {
        return findQwen3ModelPath(for: variant) != nil
    }

    // MARK: - Qwen3-VL Download

    /// Download a Qwen3-VL model from HuggingFace Hub
    public func downloadQwen3VL(
        _ model: Qwen3VLModelInfo,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        if let existingPath = Self.findQwen3VLModelPath(for: model) {
            let verification = Self.verifyShardedModel(at: existingPath)
            if verification.complete {
                progress?(1.0, "Qwen3-VL model already downloaded")
                return existingPath
            } else {
                print("Warning: Incomplete Qwen3-VL download detected. Missing files: \(verification.missing)")
                print("Re-downloading...")
            }
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        let modelUrl = try await Self.hubApi.snapshot(
            from: model.repoId,
            matching: ["*.json", "*.safetensors", "tokenizer.model"]
        ) { downloadProgress in
            let completed = downloadProgress.completedUnitCount
            let total = downloadProgress.totalUnitCount
            let fraction = downloadProgress.fractionCompleted

            let message = "Downloading file \(completed)/\(total)"
            progress?(fraction, message)
        }

        progress?(1.0, "Download complete!")
        print("Qwen3-VL model available at: \(modelUrl.path)")

        return modelUrl
    }

    /// Download a Qwen3-VL model by variant
    public func downloadQwen3VL(
        variant: Qwen3VLVariant,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        guard let model = await TextEncoderModelRegistry.shared.qwen3VLModel(withVariant: variant) else {
            throw TextEncoderModelDownloaderError.qwen3VLModelNotFound
        }
        return try await downloadQwen3VL(model, progress: progress)
    }

    /// Find a Qwen3-VL model path (checks Hub cache)
    public static func findQwen3VLModelPath(for model: Qwen3VLModelInfo) -> URL? {
        var newPath = hubDownloadDirectory
        for component in model.repoId.split(separator: "/") {
            newPath = newPath.appendingPathComponent(String(component))
        }

        if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: newPath)
            if verification.complete {
                return newPath
            }
        }

        // Check legacy Hub cache location
        let homeDir = legacyHomeDirectory
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let repoDir = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotDir = hubCache.appendingPathComponent(repoDir).appendingPathComponent("snapshots")

        if let enumerator = FileManager.default.enumerator(atPath: snapshotDir.path),
           let firstSnapshot = enumerator.nextObject() as? String {
            let snapshotPath = snapshotDir.appendingPathComponent(firstSnapshot)
            if FileManager.default.fileExists(atPath: snapshotPath.appendingPathComponent("config.json").path) {
                let verification = verifyShardedModel(at: snapshotPath)
                if verification.complete {
                    return snapshotPath
                }
            }
        }

        return nil
    }

    /// Find a Qwen3-VL model path by variant
    public static func findQwen3VLModelPath(for variant: Qwen3VLVariant) -> URL? {
        let repoId = variant.repoId

        var newPath = hubDownloadDirectory
        for component in repoId.split(separator: "/") {
            newPath = newPath.appendingPathComponent(String(component))
        }

        if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: newPath)
            if verification.complete {
                return newPath
            }
        }

        return nil
    }

    /// Check if a Qwen3-VL model is already downloaded
    public static func isQwen3VLModelDownloaded(_ model: Qwen3VLModelInfo) -> Bool {
        return findQwen3VLModelPath(for: model) != nil
    }

    /// Check if a Qwen3-VL model is already downloaded by variant
    public static func isQwen3VLModelDownloaded(variant: Qwen3VLVariant) -> Bool {
        return findQwen3VLModelPath(for: variant) != nil
    }

    // MARK: - Qwen3.5 Download

    /// Download a Qwen3.5 model from HuggingFace Hub
    public func downloadQwen35(
        _ model: Qwen35ModelInfo,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        if let existingPath = Self.findQwen35ModelPath(for: model) {
            let verification = Self.verifyShardedModel(at: existingPath)
            if verification.complete {
                progress?(1.0, "Qwen3.5 model already downloaded")
                return existingPath
            }
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")

        let modelUrl = try await Self.hubApi.snapshot(
            from: model.repoId,
            matching: ["*.json", "*.safetensors", "tokenizer.model", "*.jinja"]
        ) { downloadProgress in
            let fraction = downloadProgress.fractionCompleted
            let message = "Downloading file \(downloadProgress.completedUnitCount)/\(downloadProgress.totalUnitCount)"
            progress?(fraction, message)
        }

        progress?(1.0, "Download complete!")
        print("Qwen3.5 model available at: \(modelUrl.path)")
        return modelUrl
    }

    /// Download a Qwen3.5 model by variant
    public func downloadQwen35(
        variant: Qwen35Variant,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        guard let model = await TextEncoderModelRegistry.shared.qwen35Model(withVariant: variant) else {
            throw TextEncoderModelDownloaderError.qwen35ModelNotFound
        }
        return try await downloadQwen35(model, progress: progress)
    }

    /// Find a Qwen3.5 model path
    public static func findQwen35ModelPath(for model: Qwen35ModelInfo) -> URL? {
        var newPath = hubDownloadDirectory
        for component in model.repoId.split(separator: "/") {
            newPath = newPath.appendingPathComponent(String(component))
        }
        if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: newPath)
            if verification.complete { return newPath }
        }
        return nil
    }

    /// Find a Qwen3.5 model path by variant
    public static func findQwen35ModelPath(for variant: Qwen35Variant) -> URL? {
        var newPath = hubDownloadDirectory
        for component in variant.repoId.split(separator: "/") {
            newPath = newPath.appendingPathComponent(String(component))
        }
        if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: newPath)
            if verification.complete { return newPath }
        }
        return nil
    }

    public static func isQwen35ModelDownloaded(variant: Qwen35Variant) -> Bool {
        return findQwen35ModelPath(for: variant) != nil
    }

    /// Download a model by repo ID directly
    public func downloadByRepoId(
        _ repoId: String,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        progress?(0.0, "Starting download...")
        print("\nDownloading from HuggingFace: \(repoId)")

        let modelUrl = try await Self.hubApi.snapshot(
            from: repoId,
            matching: ["*.json", "*.safetensors"]
        )

        progress?(1.0, "Download complete!")
        print("Model available at: \(modelUrl.path)")

        return modelUrl
    }

    /// Resolve a model identifier to a local path, downloading if necessary
    public func resolveModel(
        _ identifier: String,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        // Try to find by ID
        if let model = await TextEncoderModelRegistry.shared.model(withId: identifier) {
            if let existingPath = Self.findModelPath(for: model) {
                return existingPath
            }
            return try await download(model, progress: progress)
        }

        // Try variant matching
        if let variant = ModelVariant(rawValue: identifier),
           let model = await TextEncoderModelRegistry.shared.model(withVariant: variant) {
            if let existingPath = Self.findModelPath(for: model) {
                return existingPath
            }
            return try await download(model, progress: progress)
        }

        // Check if it's a local path
        let localURL = URL(fileURLWithPath: identifier)
        if FileManager.default.fileExists(atPath: localURL.appendingPathComponent("config.json").path) {
            return localURL
        }

        // Try as a direct HuggingFace repo ID
        return try await downloadByRepoId(identifier, progress: progress)
    }

    /// Format bytes as human-readable string
    public static func formatSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

/// Errors for model downloading
public enum TextEncoderModelDownloaderError: LocalizedError {
    case modelNotFound
    case qwen3ModelNotFound
    case qwen3VLModelNotFound
    case qwen35ModelNotFound
    case downloadFailed(String)
    case invalidToken

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model not found"
        case .qwen3ModelNotFound:
            return "Qwen3 model not found"
        case .qwen3VLModelNotFound:
            return "Qwen3-VL model not found"
        case .qwen35ModelNotFound:
            return "Qwen3.5 model not found"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .invalidToken:
            return "Invalid HuggingFace token"
        }
    }
}
