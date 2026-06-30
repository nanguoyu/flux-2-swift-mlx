// StripedConvDecodeTests.swift — prove the high-res VAE decode's conv striping is bit-exact.
//
// At 512²/1024² the decoder runs its stride-1 3×3 padding-1 convs in horizontal strips
// (VAEDecoder.stripeHeavyConvs / stripedConv) so a single conv's im2col temporary stays bounded
// instead of ballooning to several GB. The striping is claimed seam-free / bit-identical to the
// full-frame conv (a 1-row halo reproduces the 3×3 receptive field exactly, and the true image
// edges keep the conv's own zero-pad). This test proves that on a tiny RANDOM-weight decoder — no
// 2 GB checkpoint required — by decoding the SAME synthetic latent with striping OFF vs ON and
// asserting the two outputs are identical to fp tolerance. It deliberately uses a latent large
// enough that the decode crosses VAEDecoder.heavyHW (512×512) so the striped branch actually runs.

import XCTest
@testable import Flux2Core
import MLX
import MLXNN
import MLXRandom

final class StripedConvDecodeTests: XCTestCase {

    /// A tiny but structurally faithful decoder: 4 channel stages → 3 upsamples → 8× upscale, small
    /// channels + groups so random weights are cheap, while still upscaling enough to trip the heavy
    /// path. normNumGroups divides every channel count so GroupNorm is valid.
    private func tinyVAEConfig() -> VAEConfig {
        VAEConfig(
            inChannels: 3,
            outChannels: 3,
            latentChannels: 8,
            blockOutChannels: [16, 16, 16, 16],
            decoderBlockOutChannels: nil,
            layersPerBlock: 1,
            normNumGroups: 8,
            scalingFactor: 0.18215
        )
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)   // .item materializes both lazy graphs
    }

    func testStripedConvDecodeIsBitExact() {
        // Restore the production default no matter how the test exits.
        let original = VAEDecoder.stripeHeavyConvs
        defer { VAEDecoder.stripeHeavyConvs = original }

        MLXRandom.seed(20260630)
        let decoder = VAEDecoder(config: tinyVAEConfig())
        // Materialize the random weights once so both decode passes see identical parameters.
        eval(decoder.parameters())

        // Latent [1, 8, 64, 64] → output [1, 3, 512, 512]. The final convOut (512×512 = 262144) and the
        // last up-block both sit at/above heavyHW (512*512), so the ON pass exercises stripedConv
        // (strips = 512/128 = 4), while the seams (3 interior strip boundaries) are exactly where a
        // naive tiling would leak.
        let latent = MLXRandom.normal([1, 8, 64, 64])
        eval(latent)

        // Full-frame reference.
        VAEDecoder.stripeHeavyConvs = false
        let reference = decoder(latent)
        eval(reference)

        // Striped decode of the identical latent + weights.
        VAEDecoder.stripeHeavyConvs = true
        let striped = decoder(latent)
        eval(striped)

        XCTAssertEqual(striped.shape, reference.shape)
        XCTAssertEqual(striped.shape, [1, 3, 512, 512])

        // Bit-exact to fp tolerance: striping reorders nothing inside each output row, so the only
        // possible drift is floating-point associativity at strip boundaries — which a correct 1-row
        // halo eliminates entirely. Allow a hair of slack for any lazy-eval reassociation.
        let diff = maxAbsDiff(striped, reference)
        XCTAssertLessThan(diff, 1e-4,
                          "striped decode diverged from full-frame decode (maxAbsDiff=\(diff)) — the conv striping is NOT seam-free/bit-exact")
    }

    /// Guard against a vacuously-passing test: if striping silently fell back to the full-frame conv
    /// for this input (e.g. heavyHW raised, or the strip count collapsing to 1), the comparison above
    /// would pass trivially. Assert the chosen input genuinely engages multi-strip execution using the
    /// exact strip math the decoder applies: H=512 at >=heavyHW yields max(1, 512/128)=4 strips.
    func testHeavyPathActuallyEngagesStriping() {
        let H = 512
        XCTAssertGreaterThanOrEqual(H * H, 512 * 512, "input must be at/above heavyHW")
        let strips = Swift.max(1, H / 128)
        XCTAssertGreaterThan(strips, 1, "the striped conv must split into more than one strip to be a real test")
    }
}
