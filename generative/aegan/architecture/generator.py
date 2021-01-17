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

        self.image = ModuleCompose(
            lambda x: x.view(-1, 16, 1, 1).expand(-1, 16, 4, 4),
            RandomFourier(16),

            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            SqueezeExcitation(32),

            ConvPixelShuffle(32, 32, upscale_factor=2),

            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            SqueezeExcitation(32),

            ConvPixelShuffle(32, 16, upscale_factor=2),

            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            SqueezeExcitation(16),

            ConvPixelShuffle(16, 1, upscale_factor=2),
            torch.tanh,
        )

    def forward(self, latent: architecture.LatentBatch) -> architecture.StandardImageBatch:
        return architecture.StandardImageBatch(
            data=self.image(latent.encoding().to(module_device(self))).cpu()
        )
