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

    /// Encode ONE reference image for streaming image-to-image — the single-image, memory-bounded form
    /// of the resident pipeline's `encodeReferenceImages`, reusing the SAME preprocess + VAE +
    /// `LatentUtils` so the streamed reference latents are byte-for-byte the facade's (the 512 parity
    /// gate checks this). `maxImageArea` caps the encoded resolution: iPhone passes 512² (≈1024 ref
    /// tokens, the streaming budget) vs the resident default 1024² (≈4096). Returns the packed reference
    /// latents `[1, refSeq, 128]` and their position ids `[refSeq, 4]` — the ids carry a distinct
    /// T-coordinate (scale 10) so the transformer separates reference tokens from the T=0 output tokens.
    /// The result is fully materialized so the caller can free the VAE before the transformer streams.
    public static func encodeReferenceImage(_ image: CGImage, maxImageArea: Int,
                                            vae: AutoencoderKLFlux2) -> (latents: MLXArray, positionIds: MLXArray) {
        let multipleOf = 32  // vae_scale_factor * 2 — same constraint the resident encoder uses
        var targetWidth = image.width
        var targetHeight = image.height
        let pixelCount = targetWidth * targetHeight
        if pixelCount > maxImageArea {
            let scale = (Double(maxImageArea) / Double(pixelCount)).squareRoot()
            targetWidth = Int(Double(targetWidth) * scale)
            targetHeight = Int(Double(targetHeight) * scale)
        }
        targetWidth = max((targetWidth / multipleOf) * multipleOf, multipleOf)
        targetHeight = max((targetHeight / multipleOf) * multipleOf, multipleOf)

        // EXACT mirror of Flux2Pipeline.encodeReferenceImages' per-image path: deterministic VAE mean
        // (samplePosterior:false), patchify, BatchNorm-normalize, pack to sequence.
        let processed = Flux2Pipeline.preprocessImageForVAE(image, targetHeight: targetHeight, targetWidth: targetWidth)
        let rawLatents = vae.encode(processed, samplePosterior: false)            // [1,32,H/8,W/8] mean
        var patchified = LatentUtils.packLatentsToPatchified(rawLatents)          // [1,128,H/16,W/16]
        patchified = LatentUtils.normalizeLatentsWithBatchNorm(
            patchified, runningMean: vae.batchNormRunningMean, runningVar: vae.batchNormRunningVar)
        let packed = LatentUtils.packPatchifiedToSequence(patchified)             // [1, refSeq, 128]
        let positionIds = LatentUtils.generateReferenceImagePositionIDs(
            latentHeights: [targetHeight / 16], latentWidths: [targetWidth / 16], scale: 10)
        // Realize the full VAE-encode chain now, while the VAE is still resident, so the caller may free
        // the VAE before the transformer streams (no encoder ↔ transformer co-residency on the phone).
        MLX.eval(packed, positionIds)
        return (latents: packed, positionIds: positionIds)
    }
}
