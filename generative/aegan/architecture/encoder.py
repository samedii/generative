import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import ModuleCompose, module_device, FunctionalBase, Tensor

from generative.aegan import architecture
from generative.tools import ConvPixelShuffle, Swish, SqueezeExcitation, RandomFourier


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent = ModuleCompose(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),  # 16x16

            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            SqueezeExcitation(16),

            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),  # 8x8

            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            SqueezeExcitation(32),

            nn.Conv2d(32, 1, 3, stride=2, padding=1, bias=False),  # 4x4

            Swish(),
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 2 * 1 * 4 * 4),  # should be global
            lambda x: x.chunk(2, dim=1),
        )

    def forward(self, standard_image: architecture.StandardImageBatch) -> architecture.LatentBatch:
        loc, log_variance = self.latent(
            standard_image.data.to(module_device(self))
        )

        return architecture.LatentBatch(
            loc=loc.cpu(),
            log_variance=log_variance.cpu(),
        )
