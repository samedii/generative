import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import ModuleCompose, module_device, FunctionalBase, Tensor

from generative.aegan import architecture
from generative.tools import Swish, SqueezeExcitation


class ImagePredictionBatch(FunctionalBase):
    logits: Tensor

    def bce_real(self):
        return F.binary_cross_entropy_with_logits(
            self.logits,
            torch.ones_like(self.logits),  # one sided smoothing?
        )

    def bce_generated(self):
        return F.binary_cross_entropy_with_logits(
            self.logits,
            torch.zeros_like(self.logits),
        )


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.real = ModuleCompose(
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
            nn.Linear(1 * 4 * 4, 1),
        )

    def forward(self, standard_image: architecture.StandardImageBatch) -> ImagePredictionBatch:
        return ImagePredictionBatch(logits=self.real(
            standard_image.data.to(module_device(self))
        ))
