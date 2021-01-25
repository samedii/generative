import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from lantern import ModuleCompose, module_device, FunctionalBase, Tensor

from generative.aegan import architecture
from generative.tools import ConvPixelShuffle, Swish, SqueezeExcitation, RandomFourier


class DecoderCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded_channels = channels * 6
        self.seq = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=5, padding=2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
            nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            SqueezeExcitation(channels),
        )

    def forward(self, x):
        return x + self.seq(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        size = 32

        self.image = ModuleCompose(
            lambda x: x.view(-1, 16, 1, 1).expand(-1, 16, 4, 4),
            RandomFourier(16),

            nn.Conv2d(16 + 16, size, kernel_size=1, bias=False),
            DecoderCell(size),
            ConvPixelShuffle(size, size, upscale_factor=2),
            DecoderCell(size),
            ConvPixelShuffle(size, size // 2, upscale_factor=2),
            DecoderCell(size // 2),
            ConvPixelShuffle(size // 2, size // 4, upscale_factor=2),
            # DecoderCell(size // 4),
            # nn.BatchNorm2d(size // 4),
            # Swish(),
            nn.Conv2d(size // 4, size // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(size // 4),
            Swish(),
            nn.Conv2d(size // 4, 1, kernel_size=3, padding=1, bias=False),
            torch.tanh,
        )

    def forward(self, latent: architecture.LatentBatch) -> architecture.StandardImageBatch:
        return architecture.StandardImageBatch(
            data=self.image(latent.encoding.to(module_device(self))).cpu()
        )
