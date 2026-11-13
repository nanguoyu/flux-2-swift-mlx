// ModelDownloader.swift - Download Flux.2 models from HuggingFace
// Copyright 2025 Vincent Gourbin

import Foundation
import CryptoKit
import FluxTextEncoders

/// Progress callback for download updates
public typealias Flux2DownloadProgressCallback = @Sendable (Double, String) -> Void

/// Downloads Flux.2 models from HuggingFace Hub
public class Flux2ModelDownloader: @unchecked Sendable {
    struct HubFile: Sendable {
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

    /// HuggingFace token for gated models
    private var hfToken: String?

    /// URLSession for downloads
    private let session: URLSession

    public convenience init(hfToken: String? = nil) {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 3600  // 1 hour for large models
        // iOS otherwise caches large downloads in URLCache, doubling on-disk usage for multi-GB
        // models (the same bug the Z-Image downloader hit). Disable it — harmless on macOS.
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.init(hfToken: hfToken, sessionConfiguration: config)
    }

    /// Test seam: the same downloader over a caller-supplied session configuration
    /// (e.g. one whose `protocolClasses` stub the network).
    init(hfToken: String? = nil, sessionConfiguration: URLSessionConfiguration) {
        self.hfToken = hfToken
        if let token = hfToken {
            setenv("HF_TOKEN", token, 1)
        }
        self.session = URLSession(configuration: sessionConfiguration)
    }

    // MARK: - Model Paths

    /// Check if a model component is downloaded
    public static func isDownloaded(_ component: ModelRegistry.ModelComponent) -> Bool {
        findModelPath(for: component) != nil
    }

    /// Find local path for a model component
    public static func findModelPath(for component: ModelRegistry.ModelComponent) -> URL? {
        // Check our local models directory
        let localPath = ModelRegistry.localPath(for: component)

        // Accept config.json OR model_index.json (Klein diffusers) OR model.safetensors.index.json
        // (mflux pre-quantized shards ship no config.json — the arch is hardcoded in Flux2Config).
        let hasConfig = FileManager.default.fileExists(atPath: localPath.appendingPathComponent("config.json").path)
        let hasModelIndex = FileManager.default.fileExists(atPath: localPath.appendingPathComponent("model_index.json").path)
        let hasShardIndex = FileManager.default.fileExists(atPath: localPath.appendingPathComponent("model.safetensors.index.json").path)

        if hasConfig || hasModelIndex || hasShardIndex {
            let verification = verifyModel(at: localPath)
            if verification.complete && verifyManifestIfPresent(at: localPath) {
                return localPath
            }
        }

        // Check configured models directory
        let repoId = repoId(for: component)
        var path = ModelRegistry.modelsDirectory

        for part in repoId.split(separator: "/") {
            path = path.appendingPathComponent(String(part))
        }

        let cacheHasConfig = FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path)
        let cacheHasModelIndex = FileManager.default.fileExists(atPath: path.appendingPathComponent("model_index.json").path)
        let cacheHasShardIndex = FileManager.default.fileExists(atPath: path.appendingPathComponent("model.safetensors.index.json").path)

        if cacheHasConfig || cacheHasModelIndex || cacheHasShardIndex {
            let verification = verifyModel(at: path)
            if verification.complete && verifyManifestIfPresent(at: path) {
                return path
            }
        }

        // Check legacy HuggingFace CLI cache (~/.cache/huggingface/hub). macOS only — iOS has no
        // user home directory, and models are downloaded fresh into the app's caches directory.
        #if os(macOS)
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
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
        let modelIndexPath = modelPath.appendingPathComponent("model_index.json")
        let shardIndexPath = modelPath.appendingPathComponent("model.safetensors.index.json")

        if FileManager.default.fileExists(atPath: configPath.path) ||
           FileManager.default.fileExists(atPath: modelIndexPath.path) ||
           FileManager.default.fileExists(atPath: shardIndexPath.path) {
            // Verify safetensors files are complete
            let verification = verifyModel(at: modelPath)
            if verification.complete && verifyManifestIfPresent(at: modelPath) {
                return modelPath
            }
        }
        #endif

        return nil
    }

    /// Get HuggingFace repo ID for a component
    private static func repoId(for component: ModelRegistry.ModelComponent) -> String {
        switch component {
        case .transformer(let variant):
            return variant.huggingFaceRepo
        case .textEncoder:
            // Text encoder uses MistralCore's download system
            return "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        case .vae(let variant):
            return variant.huggingFaceRepo
        }
    }

    /// Verify model files are complete
    public static func verifyModel(at path: URL) -> (complete: Bool, missing: [String]) {
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: path.path)) ?? []
        let safetensorsFiles = contents.filter { $0.hasSuffix(".safetensors") }

        // Single file model (various naming conventions)
        if safetensorsFiles.contains("model.safetensors") ||
           safetensorsFiles.contains("diffusion_pytorch_model.safetensors") {
            return (true, [])
        }

        // Klein bf16 models use flux-2-klein-*.safetensors naming
        if safetensorsFiles.contains(where: { $0.hasPrefix("flux-2-klein") }) {
            return (true, [])
        }

        // mflux pre-quantized sharded layout: numbered shards (0.safetensors, 1.safetensors, …)
        // enumerated in model.safetensors.index.json — no config.json, and not the model-N-of-T naming.
        // Verify via the index's weight_map: every referenced shard file must be present.
        let indexURL = path.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard let data = try? Data(contentsOf: indexURL),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = json["weight_map"] as? [String: String] else {
                return (false, ["unreadable model.safetensors.index.json"])
            }
            let requiredShards = Set(weightMap.values)
            let missingShards = requiredShards.subtracting(safetensorsFiles)
            return (missingShards.isEmpty, missingShards.sorted())
        }

        // Check for sharded model
        guard !safetensorsFiles.isEmpty else {
            return (false, ["No safetensors files found"])
        }

        // Parse sharded pattern
        var totalShards: Int?
        var foundIndices: Set<Int> = []

        for file in safetensorsFiles {
            let name = file.replacingOccurrences(of: ".safetensors", with: "")
            let parts = name.split(separator: "-")

            guard parts.count == 4,
                  parts[0] == "model",
                  parts[2] == "of",
                  let index = Int(parts[1]),
                  let total = Int(parts[3]) else {
                continue
            }

            if totalShards == nil {
                totalShards = total
            }
            foundIndices.insert(index)
        }

        if let total = totalShards {
            let expectedIndices = Set(1...total)
            let missing = expectedIndices.subtracting(foundIndices)

            if missing.isEmpty {
                return (true, [])
            } else {
                let missingFiles = missing.sorted().map {
                    "model-\(String(format: "%05d", $0))-of-\(String(format: "%05d", total)).safetensors"
                }
                return (false, missingFiles)
            }
        }

        return (true, [])
    }

    // MARK: - Download

    /// Download a model component from HuggingFace
    public func download(
        _ component: ModelRegistry.ModelComponent,
        progress: Flux2DownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = Self.findModelPath(for: component) {
            let verification = Self.verifyModel(at: existingPath)
            if verification.complete {
                progress?(1.0, "Model already downloaded")
                return existingPath
            }
        }

        let repoId = Self.repoId(for: component)
        let subfolder = Self.subfolder(for: component)
        progress?(0.0, "Fetching file list for \(component.displayName)...")

        Flux2Debug.log("Downloading \(component.displayName) from \(repoId)")

        // Get file list from HuggingFace API
        let files = try await fetchFileList(repoId: repoId, subfolder: subfolder)

        let filesToDownload = Self.componentFiles(from: files, subfolder: subfolder)

        guard !filesToDownload.isEmpty else {
            throw Flux2DownloadError.modelNotFound("No model files found in \(repoId)")
        }

        // Create destination directory
        let destDir = ModelRegistry.localPath(for: component)
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        // Names an earlier run of this downloader recorded here (possibly the recursive layout that
        // pulled sibling folders into this directory); used to clean up after success.
        let previouslyRecorded = Self.manifestFilenames(at: destDir)

        // Record the expected file set BEFORE fetching anything: until every listed file is present
        // with its expected size, findModelPath() rejects this directory, so an interrupted first
        // download can never read as installed just because one weight file already landed.
        try Self.writeManifest(filesToDownload, at: destDir)

        // Download each file
        let totalBytes = max(1, filesToDownload.reduce(Int64(0)) { $0 + ($1.size ?? 0) })
        var completedBytes: Int64 = 0

        for file in filesToDownload {
            let fileName = URL(fileURLWithPath: file.path).lastPathComponent
            let destination = destDir.appendingPathComponent(fileName)
            if Self.fileMatches(destination, expectedSize: file.size, expectedSHA256: file.sha256) {
                completedBytes += file.size ?? Self.fileSize(destination)
                progress?(min(1, Double(completedBytes) / Double(totalBytes)), "Verified \(fileName)")
                continue
            }
            progress?(Double(completedBytes) / Double(totalBytes), "Downloading \(fileName)...")

            let baseBytes = completedBytes
            let fileURL = try await downloadFile(
                repoId: repoId,
                filePath: file.path,
                to: destination,
                expectedSize: file.size,
                expectedSHA256: file.sha256,
                progress: { bytes in
                    progress?(min(1, Double(baseBytes + bytes) / Double(totalBytes)), "Downloading \(fileName)...")
                }
            )

            if let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
               let size = attrs[.size] as? Int64 {
                completedBytes += size
            }

            Flux2Debug.log("Downloaded \(fileName) (\(Self.formatSize(completedBytes)) total)")
        }

        Self.removeStaleFiles(for: component, keeping: filesToDownload, previouslyRecorded: previouslyRecorded)

        let verification = Self.verifyModel(at: destDir)
        guard verification.complete else { throw Flux2DownloadError.verificationFailed(verification.missing) }

        progress?(1.0, "Download complete: \(Self.formatSize(completedBytes))")
        return destDir
    }

    /// Get subfolder path for component within repo
    private static func subfolder(for component: ModelRegistry.ModelComponent) -> String? {
        switch component {
        case .transformer(let variant):
            return variant.huggingFaceSubfolder
        case .vae(let variant):
            return variant.huggingFaceSubfolder
        case .textEncoder:
            return nil
        }
    }

    /// The files of one component: its own folder's direct children (the repo root when
    /// `subfolder` is nil) that the loaders read. Files in nested folders are excluded: they belong
    /// to OTHER components (a Klein repo root also holds `text_encoder/`, `transformer/` and `vae/`,
    /// each with its own `config.json` / `diffusion_pytorch_model.safetensors`), the loaders only
    /// read the top level of the component directory, and flattening them into one directory made
    /// same-named files overwrite each other. This matches the layout written before listings
    /// became recursive, so existing downloads stay valid.
    static func componentFiles(from files: [HubFile], subfolder: String?) -> [HubFile] {
        let prefix = subfolder.map { $0.hasSuffix("/") ? $0 : $0 + "/" } ?? ""
        return files.filter { file in
            guard file.path.hasPrefix(prefix) else { return false }
            let relative = file.path.dropFirst(prefix.count)
            guard !relative.isEmpty, !relative.contains("/") else { return false }
            return relative.hasSuffix(".safetensors") || relative.hasSuffix(".json") || relative == "tokenizer.model"
        }
    }

    /// After a successful download, delete what this downloader owns in `component`'s directory
    /// but no longer expects: names recorded by an earlier manifest, any stray `.safetensors`, and
    /// leftover `.part` files with their `.part.validator` sidecars.
    ///
    /// Why stray `.safetensors` go: the transformer loader reads EVERY top-level `.safetensors` in
    /// the component directory, so a leftover shard (for example a text-encoder file from the old
    /// recursive layout) would be merged into the transformer weights.
    ///
    /// Why this is safe: the directory is never a caller-supplied path. It is derived from the
    /// registry (`ModelRegistry.localPath(for:)`), and nothing is touched unless that path, with
    /// symlinks resolved, lies inside `ModelRegistry.modelsDirectory`, the root this downloader
    /// creates and owns. Only regular files directly inside it are removed: symlinks (which may
    /// point anywhere) and directories are always left alone.
    static func removeStaleFiles(for component: ModelRegistry.ModelComponent, keeping files: [HubFile],
                                 previouslyRecorded: Set<String>) {
        guard let directory = managedDirectory(for: component) else { return }
        let expected = Set(files.map { URL(fileURLWithPath: $0.path).lastPathComponent })
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(atPath: directory.path)) ?? []
        for name in contents where !expected.contains(name) && name != manifestFilename {
            let owned = previouslyRecorded.contains(name)
                || name.hasSuffix(".safetensors")
                || name.hasSuffix(".part")
                || name.hasSuffix(".part.validator")
            guard owned else { continue }
            let url = directory.appendingPathComponent(name)
            // attributesOfItem does not follow a final symlink, so this sees the entry itself.
            guard let type = (try? fm.attributesOfItem(atPath: url.path))?[.type] as? FileAttributeType,
                  type == .typeRegular else { continue }
            try? fm.removeItem(at: url)
        }
    }

    /// `component`'s registry directory with symlinks resolved, or nil when it does not resolve
    /// to a directory inside the models root (for example a symlink pointing elsewhere).
    static func managedDirectory(for component: ModelRegistry.ModelComponent) -> URL? {
        let root = ModelRegistry.modelsDirectory.resolvingSymlinksInPath().standardizedFileURL
        let directory = ModelRegistry.localPath(for: component).resolvingSymlinksInPath().standardizedFileURL
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard directory.path.hasPrefix(rootPath) else { return nil }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        return directory
    }

    private static func manifestFilenames(at directory: URL) -> Set<String> {
        let url = directory.appendingPathComponent(manifestFilename)
        guard let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(DownloadManifest.self, from: data) else { return [] }
        return Set(manifest.files.map(\.filename))
    }

    /// Fetch file list from HuggingFace API
    private func fetchFileList(repoId: String, subfolder: String?) async throws -> [HubFile] {
        var urlString = "https://huggingface.co/api/models/\(repoId)/tree/main"
        if let subfolder = subfolder {
            urlString += "/\(subfolder)"
        }
        urlString += "?recursive=1&expand=1"

        guard let url = URL(string: urlString) else {
            throw Flux2DownloadError.downloadFailed("Invalid URL: \(urlString)")
        }

        var request = URLRequest(url: url)
        if let token = hfToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw Flux2DownloadError.downloadFailed("Invalid response")
        }

        if httpResponse.statusCode == 401 {
            throw Flux2DownloadError.downloadFailed(
                "Authentication required. Set HF_TOKEN environment variable or pass token to downloader."
            )
        }

        if httpResponse.statusCode == 403 {
            throw Flux2DownloadError.downloadFailed(
                "Access denied. You may need to accept the model's license at https://huggingface.co/\(repoId)"
            )
        }

        guard httpResponse.statusCode == 200 else {
            throw Flux2DownloadError.downloadFailed("HTTP \(httpResponse.statusCode)")
        }

        // Parse JSON response
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw Flux2DownloadError.downloadFailed("Invalid JSON response")
        }

        var files: [HubFile] = []
        for item in json {
            if let type = item["type"] as? String, type == "file",
               let path = item["path"] as? String {
                let size = (item["size"] as? NSNumber)?.int64Value
                let lfs = item["lfs"] as? [String: Any]
                let sha256 = lfs?["oid"] as? String
                let lfsSize = (lfs?["size"] as? NSNumber)?.int64Value
                files.append(HubFile(path: path, size: lfsSize ?? size, sha256: sha256))
            }
        }

        return files
    }

    /// Download a single file from HuggingFace
    private func downloadFile(repoId: String, filePath: String, to destination: URL,
                              expectedSize: Int64?, expectedSHA256: String?,
                              progress: @escaping @Sendable (Int64) -> Void) async throws -> URL {
        if Self.fileMatches(destination, expectedSize: expectedSize, expectedSHA256: expectedSHA256) {
            progress(expectedSize ?? Self.fileSize(destination))
            return destination
        }

        let urlString = "https://huggingface.co/\(repoId)/resolve/main/\(filePath)"

        guard let url = URL(string: urlString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? urlString) else {
            throw Flux2DownloadError.downloadFailed("Invalid URL: \(urlString)")
        }

        var request = URLRequest(url: url)
        if let token = hfToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // STREAMING download straight into `.part`: a cancel or pause keeps every received byte and
        // the next attempt resumes with Range + If-Range (see ResumableFileDownload for the
        // validator, 200/206 and Content-Range rules).
        let partURL = destination.appendingPathExtension("part")
        try await ResumableFileDownload.stream(request, into: partURL, filePath: filePath,
                                               expectedSize: expectedSize,
                                               session: session, progress: progress)

        guard Self.fileMatches(partURL, expectedSize: expectedSize, expectedSHA256: expectedSHA256) else {
            // Delete the corrupt partial (and its validator) so a retry re-downloads from scratch
            // instead of forever resuming a bad prefix that fails verification identically.
            ResumableFileDownload.discardPartial(partURL)
            throw Flux2DownloadError.downloadFailed("Hash or size verification failed for \(filePath)")
        }
        try ResumableFileDownload.commit(partURL, to: destination)

        return destination
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
        // No manifest → this model was downloaded by an older app version, by mflux, or via the HF CLI
        // cache; none of those write our manifest. Trust the structural verifyModel() that already
        // passed rather than forcing a multi-GB re-download. Likewise, a present-but-unreadable manifest
        // is not itself evidence of a bad download — only a readable manifest whose listed files fail.
        guard FileManager.default.fileExists(atPath: url.path) else { return true }
        guard let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(DownloadManifest.self, from: data) else { return true }
        for file in manifest.files {
            let path = directory.appendingPathComponent(file.filename)
            guard fileMatches(path, expectedSize: file.size, expectedSHA256: file.sha256, verifyHash: verifyHashes) else { return false }
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

    /// Download all models for a quantization configuration
    public func downloadAll(
        for config: Flux2QuantizationConfig,
        progress: Flux2DownloadProgressCallback? = nil
    ) async throws {
        let components: [ModelRegistry.ModelComponent] = [
            .transformer(ModelRegistry.TransformerVariant(rawValue: config.transformer.rawValue)!),
            .vae(.standard)
        ]

        let totalComponents = Float(components.count + 1)  // +1 for text encoder

        // Download transformer and VAE
        for (index, component) in components.enumerated() {
            let completedComponents = Float(index)
            let componentProgress: Flux2DownloadProgressCallback = { p, msg in
                let overall = (completedComponents + Float(p)) / totalComponents
                progress?(Double(overall), msg)
            }

            _ = try await download(component, progress: componentProgress)
        }

        // Text encoder is handled by MistralCore
        progress?(1.0, "All models downloaded")
    }

    // MARK: - Utilities

    /// Format bytes as human-readable string
    public static func formatSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    /// Delete a downloaded model
    public static func delete(_ component: ModelRegistry.ModelComponent) throws {
        guard let path = findModelPath(for: component) else {
            return
        }

        try FileManager.default.removeItem(at: path)
        Flux2Debug.log("Deleted \(component.displayName)")
    }

    /// Get total size of downloaded models
    public static func downloadedSize() -> Int64 {
        var total: Int64 = 0

        let components: [ModelRegistry.ModelComponent] = [
            .transformer(.qint8),
            .transformer(.bf16),
            .vae(.standard)
        ]

        for component in components {
            if let path = findModelPath(for: component) {
                total += directorySize(at: path)
            }
        }

        return total
    }

    private static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
               let size = attrs[.size] as? Int64 {
                total += size
            }
        }

        return total
    }
}

// MARK: - Errors

public enum Flux2DownloadError: LocalizedError {
    case modelNotFound(String)
    case downloadFailed(String)
    case verificationFailed([String])
    case insufficientSpace(required: Int64, available: Int64)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model not found: \(id)"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .verificationFailed(let missing):
            return "Verification failed, missing files: \(missing.joined(separator: ", "))"
        case .insufficientSpace(let required, let available):
            return "Insufficient disk space: need \(Flux2ModelDownloader.formatSize(required)), have \(Flux2ModelDownloader.formatSize(available))"
        }
    }
}
