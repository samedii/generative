import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import ModuleCompose, module_device, FunctionalBase, Tensor

from generative.aegan import architecture
from generative.tools import Swish


class LatentPredictionBatch(FunctionalBase):
    logits: Tensor

    def bce_real(self):
        return F.binary_cross_entropy_with_logits(
            self.logits,
            torch.ones_like(self.logits) * 0.95,
        )

    def bce_constructed(self):
        return F.binary_cross_entropy_with_logits(
            self.logits,
            torch.ones_like(self.logits) * 0.05,
        )


class LatentDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # transformer?

        # https://github.com/ConorLazarou/AEGAN-keras/blob/master/code/generative_model.py
        rate = 0.005
        scale = (rate / (1 - rate)) ** 0.5
        self.real = ModuleCompose(
            lambda latent: latent + torch.randn_like(latent) * 0.01,
            *[
                (
                    ModuleCompose(
                        nn.Linear(16 * (index + 1), 16),
                        lambda x: x * (1 + torch.randn_like(x) * scale),
                        nn.LayerNorm(16),
                        nn.LeakyReLU(0.02),
                    ),
                    lambda module, x: torch.cat([module(x), x], dim=1),
                )
                for index in range(16)
            ],
            nn.Linear(16 * (16 + 1), 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, 1),
        )

    def forward(self, latent: architecture.LatentBatch) -> LatentPredictionBatch:
        # These will not be the same generated as the ones used to create the images, is that a problem?
        return LatentPredictionBatch(logits=self.real(
            latent.encoding.to(module_device(self))
        ))
