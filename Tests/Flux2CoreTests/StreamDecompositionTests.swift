// StreamDecompositionTests.swift — verify the block-streaming decomposition is numerically identical
// to the monolithic transformer forward.
//
// The 1024-on-iPhone streaming path runs the transformer block-by-block via streamEmbed / runDouble /
// runSingle / streamUnembed instead of the resident callAsFunction. If that decomposition diverges
// from callAsFunction — most likely an off-by-one in the [txt;img] split at textSeqLen — the streamed
// image silently differs from the resident one and the 512 parity gate fails on-device. This test
// catches that on a tiny RANDOM-weight transformer, so it runs in CI without the 2.18 GB checkpoint.

import XCTest
@testable import Flux2Core
import MLX
import MLXRandom

final class StreamDecompositionTests: XCTestCase {

    /// A small but structurally faithful config: keeps headDim=128 + axesDimsRope=[32,32,32,32] (so
    /// RoPE is unchanged) while shrinking heads/layers/text-dim so random weights are cheap.
    private func tinyConfig() -> Flux2TransformerConfig {
        Flux2TransformerConfig(
            patchSize: 1,
            inChannels: 128,
            outChannels: 128,
            numLayers: 3,
            numSingleLayers: 4,
            attentionHeadDim: 128,
            numAttentionHeads: 2,        // innerDim = 256
            jointAttentionDim: 256,
            pooledProjectionDim: 768,
            guidanceEmbeds: false,
            axesDimsRope: [32, 32, 32, 32],
            ropeTheta: 2000.0,
            mlpRatio: 2.0,
            activationFunction: "silu"
        )
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)   // .item materializes both lazy graphs
    }

    func testStreamedDecompositionMatchesMonolithic() {
        MLXRandom.seed(20260626)
        // .disabled = no periodic eval/clearCache so the two paths share an identical op set.
        let model = Flux2Transformer2DModel(config: tinyConfig(), memoryOptimization: .disabled)

        let imgSeq = 16          // e.g. a 4×4 packed latent
        let txtSeq = 5
        let hs  = MLXRandom.normal([1, imgSeq, 128]) * 0.1
        let ehs = MLXRandom.normal([1, txtSeq, 256]) * 0.1
        let ts  = MLXArray([0.7] as [Float])
        // Position ids are [seq, 4]; exact values don't matter for parity, only that both paths share them.
        let imgIds = MLXArray((0..<(imgSeq * 4)).map { Float($0 % 7) }, [imgSeq, 4])
        let txtIds = MLXArray((0..<(txtSeq * 4)).map { Float($0 % 3) }, [txtSeq, 4])

        // Monolithic reference.
        let mono = model(hiddenStates: hs, encoderHiddenStates: ehs, timestep: ts,
                         imgIds: imgIds, txtIds: txtIds)

        // Streamed decomposition over the SAME resident blocks.
        let (h0, ctx) = model.streamEmbed(hiddenStates: hs, encoderHiddenStates: ehs,
                                          timestep: ts, imgIds: imgIds, txtIds: txtIds)
        var h = h0
        for i in 0 ..< model.doubleStreamBlockCount {
            h = Flux2Transformer2DModel.runDouble(model.transformerBlocks[i], hidden: h, context: ctx)
        }
        for j in 0 ..< model.singleStreamBlockCount {
            h = Flux2Transformer2DModel.runSingle(model.singleTransformerBlocks[j], hidden: h, context: ctx)
        }
        let streamed = model.streamUnembed(hidden: h, context: ctx)

        XCTAssertEqual(streamed.shape, mono.shape)
        let diff = maxAbsDiff(streamed, mono)
        XCTAssertLessThan(diff, 1e-4,
                          "streamed decomposition diverged from monolithic forward (maxAbsDiff=\(diff))")
    }

    /// An intentionally WRONG split point must fail, proving the test actually exercises the split
    /// (guards against a vacuously-passing parity test).
    func testOffByOneSplitIsDetected() {
        MLXRandom.seed(1)
        let model = Flux2Transformer2DModel(config: tinyConfig(), memoryOptimization: .disabled)
        let imgSeq = 16, txtSeq = 5
        let hs  = MLXRandom.normal([1, imgSeq, 128]) * 0.1
        let ehs = MLXRandom.normal([1, txtSeq, 256]) * 0.1
        let ts  = MLXArray([0.7] as [Float])
        let imgIds = MLXArray((0..<(imgSeq * 4)).map { Float($0 % 7) }, [imgSeq, 4])
        let txtIds = MLXArray((0..<(txtSeq * 4)).map { Float($0 % 3) }, [txtSeq, 4])

        let mono = model(hiddenStates: hs, encoderHiddenStates: ehs, timestep: ts,
                         imgIds: imgIds, txtIds: txtIds)
        let (h0, ctx) = model.streamEmbed(hiddenStates: hs, encoderHiddenStates: ehs,
                                          timestep: ts, imgIds: imgIds, txtIds: txtIds)
        // Corrupt the split point by one.
        let bad = Flux2Transformer2DModel.Flux2StreamContext(
            temb: ctx.temb, rope: ctx.rope, imgMod: ctx.imgMod, txtMod: ctx.txtMod,
            singleMod: ctx.singleMod, textSeqLen: ctx.textSeqLen + 1)
        var h = h0
        for i in 0 ..< model.doubleStreamBlockCount {
            h = Flux2Transformer2DModel.runDouble(model.transformerBlocks[i], hidden: h, context: bad)
        }
        for j in 0 ..< model.singleStreamBlockCount {
            h = Flux2Transformer2DModel.runSingle(model.singleTransformerBlocks[j], hidden: h, context: bad)
        }
        let streamed = model.streamUnembed(hidden: h, context: bad)
        // Shape differs (off-by-one slice) or values diverge — either way it must NOT match.
        let mismatch = streamed.shape != mono.shape || maxAbsDiff(streamed, mono) > 1e-3
        XCTAssertTrue(mismatch, "an off-by-one split should not reproduce the monolithic output")
    }
}
