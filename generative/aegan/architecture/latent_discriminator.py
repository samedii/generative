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

        # TODO: 16 layers
        # GaussianNoise, GaussianDropout
        # LayerNormalization
        # https://github.com/ConorLazarou/AEGAN-keras/blob/master/code/generative_model.py
        self.real = ModuleCompose(
            nn.Linear(1 * 4 * 4, 32),
            Swish(),
            nn.Linear(32, 32),
            Swish(),
            nn.Linear(32, 1),
        )

    def forward(self, latent: architecture.LatentBatch) -> LatentPredictionBatch:
        # These will not be the same generated as the ones used to create the images, is that a problem?
        return LatentPredictionBatch(logits=self.real(
            latent.encoding.to(module_device(self))
        ))
