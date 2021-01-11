from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli
from lantern import ModuleCompose, module_device

from generative.codebook import architecture
from generative.tools import ConvPixelShuffle


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ModuleCompose(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            F.relu,
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
        )

        self.decoder = ModuleCompose(
            ConvPixelShuffle(64, 32, upscale_factor=2),
            F.relu,
            ConvPixelShuffle(32, 1, upscale_factor=2),
            lambda x: x[:, 0],
        )

        # Alternatives:
        # - RelaxedBernoulli - maybe doesn't work?
        # - RelaxedOneHotCategorical
        # - RelaxedOneHotCategorical * Codebook

        self.image = ModuleCompose(
            self.encoder,
            lambda logits: RelaxedBernoulli(
                temperature=0.5,
                logits=logits,
            ).rsample(),
            self.decoder,
        )

    def forward(self, prepared):
        return architecture.PredictionBatch(
            images=self.image(
                prepared.to(module_device(self))
            ).cpu()
        )

    def predictions(self, feature_batch: architecture.FeatureBatch):
        return self.forward(feature_batch.stack())
