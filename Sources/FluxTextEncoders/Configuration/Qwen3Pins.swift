// Qwen3Pins.swift - Pinned HuggingFace revisions and file tables for the shipped Qwen3 encoders
//
// Each table lists exactly the files TextEncoderModelDownloader.downloadQwen3 takes: the repo's
// top-level `.json`, `.safetensors` and `tokenizer.model` files, with the size and SHA-256 they
// have at the pinned commit. Re-derive the whole table when changing a revision.

import Foundation

extension Qwen3Variant {
    /// The fixed upstream snapshot this variant downloads, or nil when the variant is unpinned
    /// (it then needs `HubDownloadPolicy.allowsUnpinnedRevisions`).
    public var pinnedSnapshot: HubSnapshotPin? {
        switch self {
        case .qwen3_4B_4bit:
            // lmstudio-community/Qwen3-4B-MLX-4bit
            return HubSnapshotPin(revision: "480ec45ecf06ae3c061e4ef11345d05c7fdb02f1", files: [
                HubPinnedFile("added_tokens.json", size: 707,
                              sha256: "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
                HubPinnedFile("config.json", size: 937,
                              sha256: "b5efdcf3b0035a3638e7228dad4d85f5c4a23f156eb7cdb0b44c8366a5d34d9b"),
                HubPinnedFile("model.safetensors", size: 2_263_022_417,
                              sha256: "365a6fff57490ca4e89e354c3295b7f2f60ac3fcb0a8a3fa06e21b80bb4d19ca"),
                HubPinnedFile("model.safetensors.index.json", size: 63_924,
                              sha256: "f7825defe5865d179c3b593173d37056be5f202dcb7153985cf74e75ecf1628b"),
                HubPinnedFile("special_tokens_map.json", size: 613,
                              sha256: "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
                HubPinnedFile("tokenizer.json", size: 11_422_654,
                              sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
                HubPinnedFile("tokenizer_config.json", size: 10_219,
                              sha256: "8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6"),
                HubPinnedFile("vocab.json", size: 2_776_833,
                              sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
            ])
        case .qwen3_4B_8bit:
            // lmstudio-community/Qwen3-4B-MLX-8bit
            return HubSnapshotPin(revision: "65ade2e73183ae9123edd7e5debabe260b21422a", files: [
                HubPinnedFile("added_tokens.json", size: 707,
                              sha256: "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
                HubPinnedFile("config.json", size: 937,
                              sha256: "15b6d29e97126abf723e3a2fad21060057ceaa677c93aaa69baf29555ac92521"),
                HubPinnedFile("model.safetensors", size: 4_274_158_989,
                              sha256: "5903c1bcd2bf5853a5d0ba7c6bb09534cc4631f5845fd5918d59d6b64e0efe50"),
                HubPinnedFile("model.safetensors.index.json", size: 63_924,
                              sha256: "f5258499bb714784358d656872d7e37c744545de99c106d0c3e75ef1aa9cab37"),
                HubPinnedFile("special_tokens_map.json", size: 613,
                              sha256: "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
                HubPinnedFile("tokenizer.json", size: 11_422_654,
                              sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
                HubPinnedFile("tokenizer_config.json", size: 10_219,
                              sha256: "8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6"),
                HubPinnedFile("vocab.json", size: 2_776_833,
                              sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
            ])
        case .qwen3_8B_8bit, .qwen3_8B_4bit:
            return nil
        }
    }
}
