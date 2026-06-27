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

        // Up blocks. At full resolution a single ResNet block holds norm1(fp32 upcast) + conv1 im2col +
        // norm2(fp32) + conv2 im2col ALL alive until the block output is evaluated — ~10GB at 1024². The
        // heavy blocks are inlined (`decodeResBlock`) so each op's scratch is freed before the next runs,
        // bounding the peak to ~one op (~2GB). Seam-free (full-frame, so GroupNorm's global spatial stats
        // are intact — spatial TILING would seam here) and bit-identical (eval only forces evaluation).
        for (resBlocks, upsample) in upBlocks {
            for resBlock in resBlocks {
                hidden = decodeResBlock(resBlock, hidden)
            }
            if let us = upsample {
                hidden = us(hidden)
                eval(hidden)
            }
        }

        // Output: the final GroupNorm + conv run at full resolution too — free the norm's fp32 scratch
        // before the convOut im2col, and the im2col before the transpose.
        hidden = convNormOut(hidden)
        hidden = silu(hidden)
        if (hidden.shape[1] * hidden.shape[2]) >= Self.heavyHW { eval(hidden) }
        hidden = convOut(hidden)
        if (hidden.shape[1] * hidden.shape[2]) >= Self.heavyHW { eval(hidden) }

        // Convert back to NCHW for output: [B, H, W, 3] -> [B, 3, H, W]
        return hidden.transposed(0, 3, 1, 2)
    }

    /// One decoder ResNet block, materializing each heavy op's scratch so it frees before the next runs.
    /// Inlines the block's submodules (rather than calling `ResnetBlock2D.callAsFunction`) so the eval
    /// barriers stay OUT of the encode/training path, where an eval inside a differentiated forward would
    /// break autograd. Below `heavyHW` it falls through to the plain block (lazy fusion, no sync cost).
    /// Bit-identical to the plain block — eval only forces evaluation, it doesn't change values.
    private func decodeResBlock(_ block: ResnetBlock2D, _ x: MLXArray) -> MLXArray {
        guard (x.shape[1] * x.shape[2]) >= Self.heavyHW else { return block(x) }
        var h = silu(block.norm1(x)); eval(h)
        h = block.conv1(h); eval(h)
        h = silu(block.norm2(h)); eval(h)
        h = block.conv2(h); eval(h)
        let shortcut = block.convShortcut.map { $0(x) } ?? x
        h = h + shortcut; eval(h)
        return h
    }
}
