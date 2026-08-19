import XCTest
@testable import Flux2Core
import FluxTextEncoders

final class KleinTextEncoderSelectionTests: XCTestCase {
    func testExplicitQuantizationSelectsExactKlein4BCheckpoint() {
        XCTAssertEqual(
            KleinTextEncoder(variant: .klein4B, quantization: .mlx4bit).resolvedQwen3Variant,
            .qwen3_4B_4bit
        )
        XCTAssertEqual(
            KleinTextEncoder(variant: .klein4B, quantization: .mlx8bit).resolvedQwen3Variant,
            .qwen3_4B_8bit
        )
    }

    func testExplicitQuantizationSelectsExactKlein9BCheckpoint() {
        XCTAssertEqual(
            KleinTextEncoder(variant: .klein9B, quantization: .mlx4bit).resolvedQwen3Variant,
            .qwen3_8B_4bit
        )
        XCTAssertEqual(
            KleinTextEncoder(variant: .klein9B, quantization: .mlx8bit).resolvedQwen3Variant,
            .qwen3_8B_8bit
        )
    }
}
