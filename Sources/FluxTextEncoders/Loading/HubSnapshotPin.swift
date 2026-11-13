// HubSnapshotPin.swift - A fixed HuggingFace revision and the exact files fetched from it
//
// Shared by Flux2ModelDownloader (Flux2Core) and TextEncoderModelDownloader (this module). A pinned
// component is listed and downloaded from `/tree/<revision>` and `/resolve/<revision>/`, and every
// file is checked against the size and SHA-256 recorded here rather than against whatever the Hub
// API reports, so a later upstream push (or a rewritten `main`) cannot change what is installed.

import Foundation
import CryptoKit

/// One file of a pinned snapshot.
public struct HubPinnedFile: Sendable, Hashable {
    /// Path relative to the repository root, as the Hub lists it (e.g. `vae/config.json`).
    public let path: String
    /// Exact byte size.
    public let size: Int64
    /// Lowercase hex SHA-256 of the whole file (for LFS files this equals the LFS oid).
    public let sha256: String

    public init(_ path: String, size: Int64, sha256: String) {
        self.path = path
        self.size = size
        self.sha256 = sha256.lowercased()
    }

    /// The name the file is stored under: components are installed into one flat directory.
    public var filename: String { URL(fileURLWithPath: path).lastPathComponent }
}

/// A repository revision (a full commit SHA) and the complete set of files one component takes
/// from it.
public struct HubSnapshotPin: Sendable, Hashable {
    public let revision: String
    public let files: [HubPinnedFile]

    public init(revision: String, files: [HubPinnedFile]) {
        self.revision = revision
        self.files = files
    }

    /// Files up to this size are hashed by `isInstalled(at:)`; larger ones are checked by size
    /// only. Weights are hashed once while downloading, never on every lookup.
    public static let installCheckHashLimit: Int64 = 1 << 20

    /// Cheap "is this component installed" check, used on every lookup: every pinned file exists
    /// in `directory` with its pinned size, and the small ones (configs, indexes) also hash to their
    /// pinned SHA-256. The directory's own download manifest is deliberately not consulted, so an
    /// install made before pinning (whose manifest came from an unpinned listing) stays valid when
    /// its files are the pinned ones. Symlinks are followed (the HF CLI cache links into blobs).
    ///
    /// A top-level `.safetensors` the pin does not name also fails the check: loaders read every
    /// top-level weight file, so a stray one (for example a text-encoder shard left by the old
    /// recursive layout) would be merged into the model. The next download removes it. Only regular
    /// files count, matching what that cleanup may delete: symlinks and directories are the user's.
    public func isInstalled(at directory: URL) -> Bool {
        let pinnedNames = Set(files.map(\.filename))
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(atPath: directory.path)) ?? []
        let hasStrayWeights = contents.contains { name in
            guard name.hasSuffix(".safetensors"), !pinnedNames.contains(name) else { return false }
            let type = (try? fm.attributesOfItem(atPath: directory.appendingPathComponent(name).path))?[.type]
            return (type as? FileAttributeType) == .typeRegular
        }
        guard !hasStrayWeights else { return false }
        return files.allSatisfy { file in
            let url = directory.appendingPathComponent(file.filename).resolvingSymlinksInPath()
            return Self.matches(url, file: file, verifyHash: file.size <= Self.installCheckHashLimit)
        }
    }

    /// Paths of pinned files a listing of the pinned revision does not contain.
    public func filesMissing(fromListing listedPaths: Set<String>) -> [String] {
        files.map(\.path).filter { !listedPaths.contains($0) }
    }

    static func matches(_ url: URL, file: HubPinnedFile, verifyHash: Bool) -> Bool {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              (attrs[.type] as? FileAttributeType) == .typeRegular,
              (attrs[.size] as? NSNumber)?.int64Value == file.size else { return false }
        guard verifyHash else { return true }
        return (try? sha256Hex(of: url)) == file.sha256
    }

    static func sha256Hex(of url: URL) throws -> String {
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
}

/// Whether components WITHOUT a pinned snapshot may be downloaded from the moving `main` branch.
///
/// Off by default: every variant OtterPix ships is pinned, and an unpinned one fails with
/// `unpinnedDownloadRefused` instead of silently following `main`. Tools that want the unpinned
/// variants (Klein 9B, Dev, Klein Base, Qwen3 8B, …) opt in by setting this to `true` before
/// downloading; this package's own CLIs and demo app do so at launch.
public enum HubDownloadPolicy {
    nonisolated(unsafe) public static var allowsUnpinnedRevisions = false
}
