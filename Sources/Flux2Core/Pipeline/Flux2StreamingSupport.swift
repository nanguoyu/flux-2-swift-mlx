import Foundation
@preconcurrency import MLX
import CoreGraphics

/// Public helpers the block-streaming engine (flux2-diffusion-engine) needs but that otherwise live
/// inside the monolithic `Flux2Pipeline`: load the VAE for a variant, and convert a VAE output tensor
/// to a `CGImage`. Both reuse the EXACT internal logic the resident pipeline uses (same VAE class,
/// config loading, weight loader, and postprocess), so the streamed path's decode stays byte-for-byte
/// identical to the resident decode — which is what the 512 parity gate checks.
public enum Flux2StreamingSupport {

    /// Load the FLUX.2 VAE for `variant` — replicates `Flux2Pipeline.loadVAE` (find path → config →
    /// build `AutoencoderKLFlux2` → load weights + batchnorm stats). Ready for `decode`/`decodeWithTiling`.
    public static func loadVAE(variant: ModelRegistry.VAEVariant) throws -> AutoencoderKLFlux2 {
        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .vae(variant)) else {
            throw Flux2Error.modelNotLoaded("VAE weights not found for variant: \(variant.rawValue)")
        }
        let vaePath = modelPath.appendingPathComponent("vae")
        let weightsPath = FileManager.default.fileExists(atPath: vaePath.path) ? vaePath : modelPath
        let configURL = weightsPath.appendingPathComponent("config.json")
        let vaeConfig: VAEConfig = FileManager.default.fileExists(atPath: configURL.path)
            ? try VAEConfig.load(from: configURL)
            : variant.vaeConfig
        let vae = AutoencoderKLFlux2(config: vaeConfig)
        let standardWeightsFile = weightsPath.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let weights = FileManager.default.fileExists(atPath: standardWeightsFile.path)
            ? try Flux2WeightLoader.loadWeights(from: standardWeightsFile)
            : try Flux2WeightLoader.loadWeights(from: weightsPath)
        try Flux2WeightLoader.applyVAEWeights(weights, to: vae)
        return vae
    }

    /// Convert a VAE decode output `[1,3,H,W]` in `[-1,1]` to a `CGImage` — the resident pipeline's
    /// exact postprocess (GPU denormalize → clip → uint8 → CGImage).
    public static func imageFromVAEOutput(_ tensor: MLXArray) -> CGImage? {
        let shape = tensor.shape
        guard shape.count == 4, shape[1] == 3 else { return nil }
        let height = shape[2], width = shape[3]
        let denormalized = (tensor + 1.0) * 127.5
        let clamped = clip(denormalized, min: 0, max: 255)
        let hwc = clamped.squeezed(axis: 0).transposed(axes: [1, 2, 0]).asType(.uint8)
        let pixelData = hwc.asArray(UInt8.self)   // materializes the graph
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                       bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
