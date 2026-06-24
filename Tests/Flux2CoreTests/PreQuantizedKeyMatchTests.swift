import XCTest
import MLX
import MLXNN
@testable import Flux2Core

/// Validates the pre-quantized 4-bit direct-load KEY MAPPING against the real
/// `mlx-community/flux2-klein-4b-4bit` checkpoint layout (387 tensors, 5 double + 20 single blocks).
///
/// This is the audit's "assert notFound == 0 against the 387-key set on macOS" gate. It builds and
/// quantizes the actual Klein 4B transformer shell and checks that every checkpoint key maps onto a
/// real model parameter AND that every quantized model layer is covered — catching the time-embedder
/// nesting bug (black output), guidance-embedder orphans, and any `quantize()`-predicate divergence,
/// all WITHOUT downloading the multi-GB checkpoint.
final class PreQuantizedKeyMatchTests: XCTestCase {

    /// The exact tensor key layout of the mflux 4-bit Klein checkpoint, reconstructed from the
    /// repo's `model.safetensors.index.json` weight_map (387 entries).
    private func checkpointKeys() -> [String] {
        var keys: [String] = []
        func quant(_ base: String) { keys += ["\(base).weight", "\(base).scales", "\(base).biases"] }
        func norm(_ base: String) { keys.append("\(base).weight") }

        // Non-block quantized linears (9 layers)
        for base in ["context_embedder", "x_embedder", "proj_out",
                     "norm_out.linear",
                     "double_stream_modulation_img.linear", "double_stream_modulation_txt.linear",
                     "single_stream_modulation.linear",
                     "time_guidance_embed.linear_1", "time_guidance_embed.linear_2"] {
            quant(base)
        }

        // Double-stream blocks (5): 12 quantized linears + 4 RMSNorms each
        for i in 0..<5 {
            let p = "transformer_blocks.\(i)"
            for l in ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
                      "attn.to_k", "attn.to_out", "attn.to_q", "attn.to_v",
                      "ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out"] {
                quant("\(p).\(l)")
            }
            for n in ["attn.norm_added_k", "attn.norm_added_q", "attn.norm_k", "attn.norm_q"] {
                norm("\(p).\(n)")
            }
        }

        // Single-stream blocks (20): 2 quantized linears + 2 RMSNorms each
        for i in 0..<20 {
            let p = "single_transformer_blocks.\(i)"
            for l in ["attn.to_out", "attn.to_qkv_mlp_proj"] { quant("\(p).\(l)") }
            for n in ["attn.norm_k", "attn.norm_q"] { norm("\(p).\(n)") }
        }

        return keys
    }

    func testKlein4B4bitKeysMapToQuantizedModelParams() {
        let keys = checkpointKeys()
        XCTAssertEqual(keys.count, 387, "checkpoint key reconstruction should total 387 tensors")

        // Build the Klein 4B transformer shell and quantize it exactly as the load path does.
        let model = Flux2Transformer2DModel(config: Flux2Model.klein4B.transformerConfig)
        quantize(model: model, groupSize: 64, bits: 4)

        var modelKeys = Set<String>()
        for (k, _) in model.parameters().flattened() { modelKeys.insert(k) }

        // Forward: every checkpoint key must map onto an existing model parameter (notFound == 0).
        var notFound: [String] = []
        var mappedSet = Set<String>()
        for key in keys {
            let mapped = Flux2WeightLoader.mapMLXQuantizedTransformerKey(key)
            mappedSet.insert(mapped)
            if !modelKeys.contains(mapped) { notFound.append("\(key) -> \(mapped)") }
        }
        XCTAssertEqual(notFound.count, 0, "unmatched checkpoint keys (first 10): \(notFound.prefix(10))")

        // Reverse: every quantized model layer (identified by `.scales`) must be filled.
        let modelScales = modelKeys.filter { $0.hasSuffix(".scales") }
        let uncovered = modelScales.filter { !mappedSet.contains($0) }
        XCTAssertEqual(uncovered.count, 0, "quant model layers with no checkpoint weight (first 10): \(uncovered.prefix(10))")

        // The two known-tricky mappings, asserted explicitly:
        // 1. the time embedder must nest under .timestepEmbedder. (not the flattened path).
        XCTAssertEqual(
            Flux2WeightLoader.mapMLXQuantizedTransformerKey("time_guidance_embed.linear_1.scales"),
            "timeGuidanceEmbed.timestepEmbedder.linear1.scales")
        XCTAssertTrue(modelKeys.contains("timeGuidanceEmbed.timestepEmbedder.linear1.scales"))
        // 2. norm_out is a quantized Linear at norm_out.linear (no adaLN half-swap on this path).
        XCTAssertTrue(modelKeys.contains("normOut.linear.scales"))
    }
}
