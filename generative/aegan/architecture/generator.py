import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from lantern import ModuleCompose, module_device, FunctionalBase, Tensor

from generative.aegan import architecture
from generative.tools import ConvPixelShuffle, Swish, SqueezeExcitation, RandomFourier


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # "channels": [256, 128, 128, 64, 64, 3],
        # "kernel_widths": [4, 4, 4, 4, 4, 4],
        # "strides": [1, 1, 1, 1, 1, 1],
        # "upsampling": [2, 2, 2, 2, 1, 1],
        # "starting_shape": [4, 4, 64],
        # "hidden_activation": "relu",
        # "output_activation": "tanh"

        channels = [256, 128, 64]
        total_channels = [128, 64 + 256, 64 + 128, 64 + 128]
        shapes = [(8, 8), (16, 16), (32, 32)]

        self.image = ModuleCompose(
            (
                ModuleCompose(
                    nn.Linear(16, 64 * 4 * 4),
                    nn.LayerNorm((64 * 4 * 4,)),
                    nn.LeakyReLU(0.02),
                    lambda x: x.view(-1, 64, 4, 4),
                ),
                lambda module, latent: (module(latent), latent),
            ),
            (
                ModuleCompose(
                    nn.Linear(16, 64),
                    nn.LayerNorm((64,)),
                    nn.LeakyReLU(0.02),
                    lambda x: x.view(-1, 64, 1, 1).expand(-1, 64, 4, 4),
                ),
                lambda module, x, latent: (x, module(latent)),
            ),
            *[
                ModuleCompose(
                    (
                        ModuleCompose(
                            lambda x, y: torch.cat([x, y], dim=1),
                            ConvPixelShuffle(from_channels, to_channels, upscale_factor=2),
                            nn.LayerNorm((to_channels, *shape)),
                            nn.LeakyReLU(0.02),
                        ),
                        lambda module, x, y: (module(x, y), y),
                    ),
                    (
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        lambda module, x, y: (x, module(y)),
                    )
                )
                for from_channels, to_channels, shape
                in zip(total_channels, channels, shapes)
            ],
            lambda x, y: torch.cat([x, y], dim=1),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            torch.tanh,
        )

    def forward(self, latent: architecture.LatentBatch) -> architecture.StandardImageBatch:
        return architecture.StandardImageBatch(
            data=self.image(latent.encoding.to(module_device(self))).cpu()
        )
