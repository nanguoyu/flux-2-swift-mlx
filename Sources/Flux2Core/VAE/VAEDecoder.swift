// VAEDecoder.swift - VAE Decoder for Flux.2
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// VAE Decoder for Flux.2
///
/// Decodes latent representations back to RGB images.
/// Supports both standard and small-decoder variants.
/// Architecture:
/// - Initial conv: latent_channels -> last_decoder_channel
/// - Mid block with attention
/// - Up blocks with ResNet blocks and upsampling
/// - Final conv: first_decoder_channel -> 3
public class VAEDecoder: Module, @unchecked Sendable {
    let config: VAEConfig

    let convIn: Conv2d
    let midBlock: (resnet1: ResnetBlock2D, attention: AttentionBlock, resnet2: ResnetBlock2D)
    let upBlocks: [(blocks: [ResnetBlock2D], upsample: Upsample2D?)]
    let convNormOut: GroupNorm
    let convOut: Conv2d

    public init(config: VAEConfig = .flux2Dev) {
        self.config = config

        // Use decoder-specific channels if available (small-decoder variant)
        let decoderChannels = config.effectiveDecoderChannels  // e.g. [96, 192, 384, 384]
        let reversedChannels = decoderChannels.reversed()

        // Initial convolution
        self.convIn = Conv2d(
            inputChannels: config.latentChannels,
            outputChannels: decoderChannels.last!,
            kernelSize: 3,
            padding: 1
        )

        // Mid block
        let midChannels = decoderChannels.last!
        self.midBlock = (
            resnet1: ResnetBlock2D(inChannels: midChannels, numGroups: config.normNumGroups),
            attention: AttentionBlock(channels: midChannels, numGroups: config.normNumGroups),
            resnet2: ResnetBlock2D(inChannels: midChannels, numGroups: config.normNumGroups)
        )

        // Up blocks (reversed order)
        var blocks: [(blocks: [ResnetBlock2D], upsample: Upsample2D?)] = []
        var prevChannels = midChannels

        for (i, outChannels) in Array(reversedChannels).enumerated() {
            var resBlocks: [ResnetBlock2D] = []

            // Diffusers adds +1 to layers_per_block for ALL up blocks in the decoder
            let numLayers = config.layersPerBlock + 1

            for j in 0..<numLayers {
                let inCh = (j == 0) ? prevChannels : outChannels
                resBlocks.append(ResnetBlock2D(
                    inChannels: inCh,
                    outChannels: outChannels,
                    numGroups: config.normNumGroups
                ))
            }
            prevChannels = outChannels

            // Upsample except for last block
            let upsample: Upsample2D?
            if i < Array(reversedChannels).count - 1 {
                upsample = Upsample2D(channels: outChannels)
            } else {
                upsample = nil
            }

            blocks.append((blocks: resBlocks, upsample: upsample))
        }
        self.upBlocks = blocks

        // Output
        self.convNormOut = GroupNorm(numGroups: config.normNumGroups, numChannels: decoderChannels[0])
        self.convOut = Conv2d(
            inputChannels: decoderChannels[0],
            outputChannels: config.outChannels,
            kernelSize: 3,
            padding: 1
        )
    }

    /// Spatial-area threshold above which the decode materializes each heavy op individually. Catches
    /// the 512² and 1024² decoder stages (where a single conv's im2col temporary is ~1.7GB in fp16);
    /// lower-res stages stay below it and keep MLX's lazy fusion (no sync cost).
    private static let heavyHW = 512 * 512

    /// Decode-time conv striping (see `stripedConv`). Public so tools can flip it OFF to capture a
    /// full-frame reference and prove the striped decode is bit-identical. Production = on.
    nonisolated(unsafe) public static var stripeHeavyConvs = true

    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        // z shape: [B, latent_channels, H/8, W/8] (NCHW from transformer)
        // Convert to NHWC for MLX Conv2d
        var hidden = z.transposed(0, 2, 3, 1)  // [B, H/8, W/8, latent_channels]

        // Initial conv
        hidden = convIn(hidden)

        // Mid block
        hidden = midBlock.resnet1(hidden)
        hidden = midBlock.attention(hidden)
        hidden = midBlock.resnet2(hidden)
        eval(hidden)

        // Up blocks. At full resolution the cost is a SINGLE op: one 3×3 conv's im2col temporary is
        // ~7GB at 1024² (e.g. the 192→96 conv) — eval can't shrink it, so the heavy convs are run in
        // horizontal strips (`decodeConv`/`stripedConv`, exact via a 1-row halo) and each op's scratch
        // is materialized before the next. Seam-free: GroupNorm's global spatial stats stay full-frame
        // (spatial TILING would seam there); only the spatially-local convs are split.
        for (resBlocks, upsample) in upBlocks {
            for resBlock in resBlocks {
                hidden = decodeResBlock(resBlock, hidden)
            }
            if let us = upsample {
                hidden = decodeConv(us.conv, us.upsampleNearest(hidden))
                eval(hidden)
            }
        }

        // Output: GroupNorm stays full-frame (global stats); the final conv is striped like the rest.
        hidden = convNormOut(hidden)
        hidden = silu(hidden)
        if (hidden.shape[1] * hidden.shape[2]) >= Self.heavyHW { eval(hidden) }
        hidden = decodeConv(convOut, hidden)

        // Convert back to NCHW for output: [B, H, W, 3] -> [B, 3, H, W]
        return hidden.transposed(0, 3, 1, 2)
    }

    /// One decoder ResNet block at full resolution, with each heavy conv striped and each op's scratch
    /// materialized so it frees before the next runs. Inlines the block's submodules (rather than
    /// `ResnetBlock2D.callAsFunction`) so the eval barriers stay OUT of the encode/training path, where
    /// an eval inside a differentiated forward would break autograd. Below `heavyHW` it falls through to
    /// the plain block. Bit-identical to the plain block.
    private func decodeResBlock(_ block: ResnetBlock2D, _ x: MLXArray) -> MLXArray {
        guard (x.shape[1] * x.shape[2]) >= Self.heavyHW else { return block(x) }
        var h = silu(block.norm1(x)); eval(h)
        h = decodeConv(block.conv1, h)
        h = silu(block.norm2(h)); eval(h)
        h = decodeConv(block.conv2, h)
        let shortcut = block.convShortcut.map { $0(x) } ?? x
        h = h + shortcut; eval(h)
        return h
    }

    /// Run a stride-1 3×3 padding-1 conv. Full-frame below `heavyHW` or when striping is disabled;
    /// otherwise striped into ~128-row horizontal bands.
    private func decodeConv(_ conv: Conv2d, _ x: MLXArray) -> MLXArray {
        let big = (x.shape[1] * x.shape[2]) >= Self.heavyHW
        guard big && Self.stripeHeavyConvs else { return conv(x) }
        return stripedConv(conv, x, strips: Swift.max(1, x.shape[1] / 128))
    }

    /// Compute a stride-1, 3×3, padding-1 conv in horizontal strips with a 1-row halo, so the im2col
    /// temporary is bounded to one strip instead of the full-resolution map (~7GB at 1024²). EXACT: the
    /// 3×3 receptive field reaches exactly 1 row beyond each output row, so feeding rows [a-1 … b+1] and
    /// cropping the halo output row(s) reproduces the full conv bit-for-bit — and the true top/bottom
    /// image edges keep the conv's own zero-pad (the strip there starts/ends at the real edge). Only
    /// valid for the decoder's stride-1 3×3 padding-1 convs.
    private func stripedConv(_ conv: Conv2d, _ x: MLXArray, strips: Int) -> MLXArray {
        guard strips > 1 else { return conv(x) }
        let H = x.shape[1]
        let stripH = (H + strips - 1) / strips
        var outs: [MLXArray] = []
        var a = 0
        while a < H {
            let b = Swift.min(a + stripH, H) - 1          // this strip emits output rows [a ... b]
            let inStart = Swift.max(0, a - 1)             // include 1 halo row above (unless top edge)
            let inEnd = Swift.min(H, b + 2)               // exclusive; 1 halo row below + 1 for exclusivity
            let strip = x[0..., inStart ..< inEnd, 0..., 0...]
            let convd = conv(strip)                        // padding=1 → same row count as `strip`
            let cropTop = a - inStart                      // 1 for interior strips, 0 at the true top edge
            let outStrip = convd[0..., cropTop ..< (cropTop + (b - a + 1)), 0..., 0...]
            eval(outStrip)
            outs.append(outStrip)
            a = b + 1
        }
        return concatenated(outs, axis: 1)
    }
}
