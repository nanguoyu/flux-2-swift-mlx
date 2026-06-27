// AutoencoderKL.swift - Flux.2 VAE (AutoencoderKL)
// Copyright 2025 Vincent Gourbin

import Foundation
import CoreGraphics
import MLX
import MLXNN
import MLXRandom

/// Configuration for spatial tiling during VAE decoding
/// Used to decode large images in tiles to reduce peak memory usage
public struct VAETilingConfig: Sendable {
    /// Tile size in latent space (e.g., 64 = 512px after 8x upscale)
    public var tileSize: Int

    /// Overlap between tiles in latent space for blending (e.g., 8 = 64px)
    public var tileOverlap: Int

    /// Minimum image dimension (in latent space) to trigger tiling
    /// Images smaller than this are decoded without tiling
    public var minTileThreshold: Int

    public init(tileSize: Int = 64, tileOverlap: Int = 8, minTileThreshold: Int = 128) {
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.minTileThreshold = minTileThreshold
    }

    /// Default tiling config (tiles of 512px with 64px overlap)
    public static let `default` = VAETilingConfig()

    /// Aggressive tiling for very large images (tiles of 256px)
    public static let aggressive = VAETilingConfig(tileSize: 32, tileOverlap: 4, minTileThreshold: 64)

    /// Disabled - no tiling
    public static let disabled = VAETilingConfig(tileSize: 9999, tileOverlap: 0, minTileThreshold: 9999)
}

/// Variational Autoencoder for Flux.2
///
/// Key differences from Flux.1 VAE:
/// - 32 latent channels (vs 4)
/// - Uses BatchNorm2d for latent normalization
/// - Patch size (2, 2) for latent packing
/// - Supports spatial tiling for large image decoding
public class AutoencoderKLFlux2: Module, @unchecked Sendable {
    let config: VAEConfig

    let encoder: VAEEncoder
    let decoder: VAEDecoder

    // Quantization convolutions (optional)
    let quantConv: Conv2d?
    let postQuantConv: Conv2d?

    // BatchNorm for latent normalization (Flux.2 specific)
    let latentBatchNorm: BatchNorm2d

    /// Scaling factor for latent space
    let scalingFactor: Float

    public init(config: VAEConfig = .flux2Dev) {
        self.config = config
        self.scalingFactor = config.scalingFactor

        self.encoder = VAEEncoder(config: config)
        self.decoder = VAEDecoder(config: config)

        // Optional quant convs (some models use them)
        self.quantConv = Conv2d(
            inputChannels: config.latentChannels * 2,
            outputChannels: config.latentChannels * 2,
            kernelSize: 1
        )
        self.postQuantConv = Conv2d(
            inputChannels: config.latentChannels,
            outputChannels: config.latentChannels,
            kernelSize: 1
        )

        // BatchNorm for latent normalization
        self.latentBatchNorm = BatchNorm2d(numFeatures: config.latentChannels)
    }

    /// Encode image to latent space
    /// - Parameters:
    ///   - x: Input image [B, 3, H, W] normalized to [-1, 1]
    ///   - samplePosterior: Whether to sample from posterior or use mean
    /// - Returns: Latent representation [B, 32, H/8, W/8]
    public func encode(_ x: MLXArray, samplePosterior: Bool = true) -> MLXArray {
        var h = encoder(x)  // Output is NCHW

        // Apply quant conv if present - needs NHWC
        if let qc = quantConv {
            let hNHWC = h.transposed(0, 2, 3, 1)  // NCHW -> NHWC
            let qcOut = qc(hNHWC)
            h = qcOut.transposed(0, 3, 1, 2)  // NHWC -> NCHW
        }

        // Sample from posterior (NCHW format)
        let mean = h[0..., 0..<config.latentChannels, 0..., 0...]
        let logVar = h[0..., config.latentChannels..., 0..., 0...]

        var latent: MLXArray
        if samplePosterior {
            let std = exp(0.5 * logVar)
            let noise = MLXRandom.normal(mean.shape)
            latent = mean + std * noise
        } else {
            latent = mean
        }

        // NOTE: BatchNorm is NOT applied during encode for Flux.2
        // The BatchNorm weights are for patchified format (128 channels)
        // and encode produces unpatchified latents (32 channels).
        // Normalization is handled later in the pipeline after patchifying.

        // NOTE: Flux.2 VAE does NOT apply scaling factor during encode!
        // (Standard SD VAE uses scalingFactor=0.18215, but Flux.2 does not)
        // See: diffusers AutoencoderKL.encode() for Flux.2

        return latent
    }

    /// Decode latent to image
    /// - Parameter z: Latent representation [B, 32, H/8, W/8] (NCHW format)
    /// - Returns: Decoded image [B, 3, H, W] in [-1, 1] (NCHW format)
    /// - Note: Flux.2 VAE does NOT use scaling factor (unlike SD VAE)
    public func decode(_ z: MLXArray) -> MLXArray {
        // NOTE: Flux.2 VAE does NOT apply scaling factor!
        // (Standard SD VAE uses scalingFactor=0.18215, but Flux.2 does not)
        var latent = z

        // Apply post-quant conv if present - needs NHWC
        if let pqc = postQuantConv {
            let latentNHWC = latent.transposed(0, 2, 3, 1)  // NCHW -> NHWC
            let pqcOut = pqc(latentNHWC)
            latent = pqcOut.transposed(0, 3, 1, 2)  // NHWC -> NCHW
        }

        // Decode - decoder handles NCHW->NHWC->NCHW internally
        return decoder(latent)
    }

    /// Full forward pass (for training/debugging)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let z = encode(x)
        return decode(z)
    }

    // MARK: - Tiled Decoding

    /// Decode latent to image with optional spatial tiling
    /// Use this for large images to reduce peak memory usage
    /// - Parameters:
    ///   - z: Latent representation [B, 32, H/8, W/8] (NCHW format)
    ///   - tiling: Tiling configuration (nil = auto-detect based on size)
    /// - Returns: Decoded image [B, 3, H, W] in [-1, 1] (NCHW format)
    public func decodeWithTiling(_ z: MLXArray, tiling: VAETilingConfig? = nil) -> MLXArray {
        let config = tiling ?? .default
        let H = z.shape[2]
        let W = z.shape[3]

        // Check if tiling is needed
        if H <= config.minTileThreshold && W <= config.minTileThreshold {
            Flux2Debug.verbose("VAE: Image small enough, decoding without tiling")
            return decode(z)
        }

        Flux2Debug.log("VAE: Using tiled decoding for \(H*8)x\(W*8) image")
        return decodeTiled(z, tileSize: config.tileSize, overlap: config.tileOverlap)
    }

    /// Tiled decoding (blend-accumulate). Decodes the latent in overlapping spatial tiles and blends
    /// them into a full-size accumulator with boundary-aware feathered weights. This bounds the peak to
    /// ~ONE tile's decode (the decoder's high-res feature maps are the 1024 memory wall, ~8.7GB full),
    /// yields the EXACTLY correct output size (the old fixed-overlap crop gave 1152px on clamped edge
    /// tiles), and feathers only interior edges so the image border isn't divided by a near-zero weight.
    private func decodeTiled(_ z: MLXArray, tileSize: Int, overlap: Int) -> MLXArray {
        let scale = 8
        let H = z.shape[2], W = z.shape[3]
        let outH = H * scale, outW = W * scale
        let stride = Swift.max(1, tileSize - overlap)

        func starts(_ total: Int) -> [Int] {
            if total <= tileSize { return [0] }
            var s: [Int] = []
            var p = 0
            while p + tileSize < total { s.append(p); p += stride }
            s.append(total - tileSize)            // last tile clamped to the end
            return s
        }
        let ys = starts(H), xs = starts(W)
        let outTile = tileSize * scale
        let outOverlap = overlap * scale

        // Full-size accumulators (weighted output sum + weight sum) — small; one tile decode at a time.
        var acc = MLXArray.zeros([1, 3, outH, outW])
        var wsum = MLXArray.zeros([1, 1, outH, outW])

        for (iy, y0) in ys.enumerated() {
            for (ix, x0) in xs.enumerated() {
                let tile = z[0..., 0..., y0 ..< (y0 + tileSize), x0 ..< (x0 + tileSize)]
                let decoded = decode(tile)                              // [1, 3, outTile, outTile]
                let mask = blendMask(size: outTile, overlap: outOverlap,
                                     top: iy > 0, bottom: iy < ys.count - 1,
                                     left: ix > 0, right: ix < xs.count - 1)
                let oy = y0 * scale, ox = x0 * scale
                let widths: [IntOrPair] = [[0, 0], [0, 0],
                                           [oy, outH - oy - outTile], [ox, outW - ox - outTile]]
                acc = acc + padded(decoded * mask, widths: widths)
                wsum = wsum + padded(mask, widths: widths)
                eval(acc, wsum)                                         // free this tile's feature maps
                MLX.Memory.clearCache()
            }
        }
        let result = acc / maximum(wsum, MLXArray(Float(1e-6)))
        Flux2Debug.log("VAE: Tiled decoding completed, output shape: \(result.shape)")
        return result
    }

    /// Boundary-aware feathered blend weights: 1 in the interior, linear falloff to ~0 ONLY on edges
    /// flagged as overlapping a neighbor (image-border edges stay 1 — the accumulator divides by the
    /// weight sum, and a near-zero border weight would blow up).
    private func blendMask(size: Int, overlap: Int, top: Bool, bottom: Bool, left: Bool, right: Bool) -> MLXArray {
        let ov = Swift.max(1, overlap)
        var w = [Float](repeating: 1.0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                var weight: Float = 1.0
                if top, y < ov { weight *= (Float(y) + 0.5) / Float(ov) }
                if bottom, y >= size - ov { weight *= (Float(size - 1 - y) + 0.5) / Float(ov) }
                if left, x < ov { weight *= (Float(x) + 0.5) / Float(ov) }
                if right, x >= size - ov { weight *= (Float(size - 1 - x) + 0.5) / Float(ov) }
                w[y * size + x] = Swift.max(weight, 1e-3)
            }
        }
        return MLXArray(w).reshaped([1, 1, size, size])
    }

    /// Create a blending weight mask for tile overlap
    private func createBlendMask(size: Int, overlap: Int) -> MLXArray {
        // Create linear ramp weights for blending
        var weights = [Float](repeating: 1.0, count: size * size)

        // Apply linear falloff in overlap regions
        for y in 0..<size {
            for x in 0..<size {
                var weight: Float = 1.0

                // Top edge
                if y < overlap {
                    weight *= Float(y) / Float(overlap)
                }
                // Bottom edge
                if y >= size - overlap {
                    weight *= Float(size - 1 - y) / Float(overlap)
                }
                // Left edge
                if x < overlap {
                    weight *= Float(x) / Float(overlap)
                }
                // Right edge
                if x >= size - overlap {
                    weight *= Float(size - 1 - x) / Float(overlap)
                }

                weights[y * size + x] = weight
            }
        }

        return MLXArray(weights).reshaped([1, 1, size, size])
    }
}

// MARK: - Latent Packing/Unpacking

extension AutoencoderKLFlux2 {
    /// Pack latents for transformer input
    /// Converts [B, C, H, W] to [B, H*W, C] with patch handling
    /// - Parameters:
    ///   - latents: Encoded latents [B, 32, H, W]
    ///   - patchSize: Patch size (2, 2) for Flux.2
    /// - Returns: Packed latents [B, (H/p)*(W/p), C*p*p]
    public func packLatents(_ latents: MLXArray, patchSize: (Int, Int) = (2, 2)) -> MLXArray {
        let shape = latents.shape
        let B = shape[0]
        let C = shape[1]
        let H = shape[2]
        let W = shape[3]

        let pH = patchSize.0
        let pW = patchSize.1

        // Reshape to patches
        // [B, C, H, W] -> [B, C, H/pH, pH, W/pW, pW]
        var packed = latents.reshaped([B, C, H / pH, pH, W / pW, pW])

        // Permute to [B, H/pH, W/pW, C, pH, pW]
        packed = packed.transposed(0, 2, 4, 1, 3, 5)

        // Flatten patches and spatial dims
        // [B, H/pH, W/pW, C, pH, pW] -> [B, (H/pH)*(W/pW), C*pH*pW]
        let numPatches = (H / pH) * (W / pW)
        let patchDim = C * pH * pW

        packed = packed.reshaped([B, numPatches, patchDim])

        return packed
    }

    /// Unpack latents from transformer output
    /// Converts [B, H*W, C] back to [B, C, H, W]
    /// - Parameters:
    ///   - packed: Packed latents [B, (H/p)*(W/p), C*p*p]
    ///   - height: Target height (after unpacking)
    ///   - width: Target width (after unpacking)
    ///   - patchSize: Patch size (2, 2) for Flux.2
    /// - Returns: Unpacked latents [B, C, H, W]
    public func unpackLatents(
        _ packed: MLXArray,
        height: Int,
        width: Int,
        patchSize: (Int, Int) = (2, 2)
    ) -> MLXArray {
        let shape = packed.shape
        let B = shape[0]

        let pH = patchSize.0
        let pW = patchSize.1

        let outH = height / 8  // VAE downsamples by 8
        let outW = width / 8

        let numPatchesH = outH / pH
        let numPatchesW = outW / pW
        let C = config.latentChannels

        // [B, numPatches, C*pH*pW] -> [B, numPatchesH, numPatchesW, C, pH, pW]
        var unpacked = packed.reshaped([B, numPatchesH, numPatchesW, C, pH, pW])

        // Permute to [B, C, numPatchesH, pH, numPatchesW, pW]
        unpacked = unpacked.transposed(0, 3, 1, 4, 2, 5)

        // Reshape to final [B, C, H, W]
        unpacked = unpacked.reshaped([B, C, outH, outW])

        return unpacked
    }
}

// MARK: - Weight Loading

extension AutoencoderKLFlux2 {
    /// Load weights from safetensors
    public func loadWeights(from url: URL) throws {
        // Implementation in WeightLoader.swift
        fatalError("Weight loading not yet implemented")
    }

    /// Load BatchNorm running statistics
    public func loadBatchNormStats(runningMean: MLXArray, runningVar: MLXArray) {
        latentBatchNorm.runningMean = runningMean
        latentBatchNorm.runningVar = runningVar
    }

    /// Get BatchNorm running mean for latent normalization
    public var batchNormRunningMean: MLXArray {
        latentBatchNorm.runningMean
    }

    /// Get BatchNorm running variance for latent normalization
    public var batchNormRunningVar: MLXArray {
        latentBatchNorm.runningVar
    }
}

// MARK: - Image Processing Utilities

/// Process image for VAE input
/// - Parameter image: CGImage to process
/// - Returns: MLXArray normalized to [-1, 1], shape [1, 3, H, W]
public func preprocessImageForVAE(_ image: CGImage) -> MLXArray {
    let width = image.width
    let height = image.height

    // Get pixel data
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          ) else {
        fatalError("Failed to create graphics context")
    }

    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let data = context.data else {
        fatalError("Failed to get pixel data")
    }

    let pixelData = data.assumingMemoryBound(to: UInt8.self)
    var floatData: [Float] = []

    // Convert RGBA to RGB and normalize to [-1, 1]
    for y in 0..<height {
        for x in 0..<width {
            let offset = (y * width + x) * 4
            floatData.append(Float(pixelData[offset]) / 127.5 - 1.0)      // R
            floatData.append(Float(pixelData[offset + 1]) / 127.5 - 1.0)  // G
            floatData.append(Float(pixelData[offset + 2]) / 127.5 - 1.0)  // B
        }
    }

    // Create MLXArray [H, W, 3] and permute to [1, 3, H, W]
    let array = MLXArray(floatData).reshaped([height, width, 3])
    return array.transposed(2, 0, 1).expandedDimensions(axis: 0)
}

/// Convert VAE output to CGImage
/// - Parameter tensor: MLXArray in [-1, 1], shape [1, 3, H, W]
/// - Returns: CGImage
public func postprocessVAEOutput(_ tensor: MLXArray) -> CGImage? {
    // Remove batch dim and permute to [H, W, 3]
    let image = tensor.squeezed(axis: 0).transposed(1, 2, 0)

    // Clamp and convert to [0, 255]
    let clamped = clip(image, min: -1, max: 1)
    let normalized = ((clamped + 1) * 127.5).asType(.uint8)

    // Get dimensions
    let shape = normalized.shape
    let height = shape[0]
    let width = shape[1]

    // Evaluate and get data
    eval(normalized)
    let data = normalized.asArray(UInt8.self)

    // Create CGImage
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let provider = CGDataProvider(data: Data(data) as CFData) else {
        return nil
    }

    return CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 24,
        bytesPerRow: width * 3,
        space: colorSpace,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    )
}
