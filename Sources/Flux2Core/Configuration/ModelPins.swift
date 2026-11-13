// ModelPins.swift - Pinned HuggingFace revisions and file tables for the shipped components
//
// Each table lists exactly the files Flux2ModelDownloader takes for that component (the direct
// children of its folder that are `.safetensors`, `.json` or `tokenizer.model`), with the size and
// SHA-256 they have at the pinned commit. Changing a revision means re-deriving the whole table
// from `/api/models/<repo>/tree/<revision>?recursive=1&expand=1` (LFS oid + size) and hashing the
// small non-LFS files, then updating the tests that pin these values.

import Foundation
import FluxTextEncoders

extension ModelRegistry.TransformerVariant {
    /// The fixed upstream snapshot this variant downloads, or nil when the variant is unpinned
    /// (it then needs `HubDownloadPolicy.allowsUnpinnedRevisions`).
    public var pinnedSnapshot: HubSnapshotPin? {
        switch self {
        case .klein4B_bf16:
            // black-forest-labs/FLUX.2-klein-4B, distilled bf16 transformer at the repo root.
            return HubSnapshotPin(revision: "e7b7dc27f91deacad38e78976d1f2b499d76a294", files: [
                HubPinnedFile("flux-2-klein-4b.safetensors", size: 7_751_105_712,
                              sha256: "ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343"),
                HubPinnedFile("model_index.json", size: 446,
                              sha256: "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0"),
            ])
        case .klein4B_8bit:
            // aydin99/FLUX.2-klein-4B-int8 (community quanto int8 quantization).
            return HubSnapshotPin(revision: "f38839c94b597211dea83b1326f1bb4ce4142524", files: [
                HubPinnedFile("config.json", size: 523,
                              sha256: "8e41d76ed6ff0ff2238ee5002579359967ddda011e5a5b448e700363909c9330"),
                HubPinnedFile("diffusion_pytorch_model.safetensors", size: 3_877_645_084,
                              sha256: "fb6e95fd62f2af89151cef9f22a99da842a0968ecec35bdeaf8d58403ad14f8d"),
                HubPinnedFile("quanto_qmap.json", size: 12_269,
                              sha256: "df61c076f376e322e7ba54805473e9d6d1aaa3f714d146f589cf2859953e3db8"),
            ])
        case .klein4B_4bit:
            // mlx-community/flux2-klein-4b-4bit, mflux pre-quantized shards under transformer/.
            return HubSnapshotPin(revision: "860e87183ceb29e39627c0612ebd66d8ea66e68c", files: [
                HubPinnedFile("transformer/0.safetensors", size: 2_145_323_732,
                              sha256: "3a75fc123a9ceb583b0d0202cc37166da3e1fc0ac8cd4a430987ceaf8e845b27"),
                HubPinnedFile("transformer/1.safetensors", size: 34_727_761,
                              sha256: "06998c075c1d9543f17f1cc7ae7170c365cdb3c09a1f78056804a252bb168fd5"),
                HubPinnedFile("transformer/model.safetensors.index.json", size: 26_945,
                              sha256: "dc46409ae5d1a10ad3c5b87c615efd1d23e4b9f1c5caf46b8d92e459dadc48b9"),
            ])
        case .bf16, .qint8, .klein4B_base_bf16, .klein9B_bf16, .klein9B_base_bf16, .klein9B_kv_bf16:
            return nil
        }
    }
}

extension ModelRegistry.VAEVariant {
    /// The fixed upstream snapshot this VAE downloads.
    public var pinnedSnapshot: HubSnapshotPin? {
        switch self {
        case .standard:
            // black-forest-labs/FLUX.2-klein-4B, vae/ folder (same commit as the bf16 transformer).
            return HubSnapshotPin(revision: "e7b7dc27f91deacad38e78976d1f2b499d76a294", files: [
                HubPinnedFile("vae/config.json", size: 821,
                              sha256: "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948"),
                HubPinnedFile("vae/diffusion_pytorch_model.safetensors", size: 168_120_878,
                              sha256: "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
            ])
        case .smallDecoder:
            // black-forest-labs/FLUX.2-small-decoder, files at the repo root.
            return HubSnapshotPin(revision: "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a", files: [
                HubPinnedFile("config.json", size: 842,
                              sha256: "88641eb0ebe46877b32822fb91aff828574765ef6b2b4bbc73d379051d006267"),
                HubPinnedFile("diffusion_pytorch_model.safetensors", size: 249_521_340,
                              sha256: "d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b"),
                HubPinnedFile("full_encoder_small_decoder.safetensors", size: 249_519_092,
                              sha256: "ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62"),
                HubPinnedFile("small_decoder.safetensors", size: 111_801_980,
                              sha256: "ad783347747561d974f347f1b7377ebca15330621ace6a2e3f8dfaa83f5c211f"),
            ])
        }
    }
}

extension ModelRegistry.ModelComponent {
    /// The fixed upstream snapshot for this component, or nil when it is unpinned. The Mistral
    /// text encoder is not downloaded by Flux2ModelDownloader and is never pinned here.
    public var pinnedSnapshot: HubSnapshotPin? {
        switch self {
        case .transformer(let variant): return variant.pinnedSnapshot
        case .vae(let variant): return variant.pinnedSnapshot
        case .textEncoder: return nil
        }
    }
}
