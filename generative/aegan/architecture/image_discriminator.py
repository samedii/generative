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
            torch.ones_like(self.logits) * 0.95,  # one sided smoothing?
        )

    def bce_generated(self):
        return F.binary_cross_entropy_with_logits(
            self.logits,
            torch.ones_like(self.logits) * 0.05,
        )


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        channels = [1, 64, 64, 128, 128, 128, 256]
        kernel_sizes = [3, 3, 3, 3, 3, 3]
        strides = [2, 2, 2, 1, 1, 1]
        shapes = [(16, 16), (8, 8), (4, 4), (4, 4), (4, 4), (4, 4)]

        self.real = ModuleCompose(
            *[
                ModuleCompose(
                    nn.Conv2d(from_channels, to_channels, kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.LayerNorm((to_channels, *shape)),
                    nn.LeakyReLU(0.02),
                )
                for from_channels, to_channels, kernel_size, stride, shape
                in zip(channels[:-1], channels[1:], kernel_sizes, strides, shapes)
            ],
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
        )

    def forward(self, standard_image: architecture.StandardImageBatch) -> ImagePredictionBatch:
        return ImagePredictionBatch(logits=self.real(
            standard_image.data.to(module_device(self))
        ))
