import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import ModuleCompose, module_device, FunctionalBase, Tensor

from generative.aegan import architecture
from generative.tools import ConvPixelShuffle, Swish, SqueezeExcitation, RandomFourier


class EncoderCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(dim),
            Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SqueezeExcitation(dim),
        )

    def forward(self, x):
        return x + self.seq(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        size = 32

        self.latent = ModuleCompose(
            nn.Conv2d(1, size, 3, stride=2, padding=1, bias=False),  # 16x16
            EncoderCell(size),
            nn.Conv2d(size, 2 * size, 3, stride=2, padding=1, bias=False),  # 8x8
            EncoderCell(2 * size),
            nn.Conv2d(2 * size, 1, 3, stride=2, padding=1, bias=False),  # 4x4
            Swish(),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 1 * 4 * 4),
        )

    def forward(self, standard_image: architecture.StandardImageBatch) -> architecture.LatentBatch:
        return architecture.LatentBatch(encoding=self.latent(
            standard_image.data.to(module_device(self))
        ).cpu())
