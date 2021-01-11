from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from lantern import ModuleCompose, module_device

from generative.vae import architecture
from generative.tools import ConvPixelShuffle, Swish, SqueezeExcitation, RandomFourier


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ModuleCompose(
            # nn.Flatten(),
            # nn.Linear(32 * 32, 400),
            # F.relu,
            # nn.Linear(400, 20 * 2),

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

            # nn.Conv2d(channels, channels, kernel_size=1),
            # module.Swish(),
            # nn.Conv2d(channels, latent_channels * 2, kernel_size=1),
            # partial(torch.chunk, chunks=2, dim=1),

            # nn.BatchNorm2d(64),
            Swish(),
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 2 * 1 * 4 * 4),  # should be global
            lambda x: x.chunk(2, dim=1),
        )

        self.decoder = ModuleCompose(
            # nn.Linear(20, 400),
            # F.relu,
            # nn.Linear(400, 32 * 32),
            # lambda x: x.view(-1, 32, 32),
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
            torch.sigmoid,
            lambda x: x[:, 0],
        )

    def forward(self, prepared):
        loc, log_variance = self.encoder(
            prepared.to(module_device(self))
        )

        # Alternatives:
        # - RelaxedBernoulli
        # - RelaxedCategorical
        # - RelaxedCategorical * Codebook

        encoding = loc + torch.randn_like(loc) * torch.exp(0.5 * log_variance)

        return architecture.PredictionBatch(
            images=self.decoder(encoding).cpu(),
            loc=loc.cpu(),
            log_variance=log_variance.cpu(),
            encoding=encoding.cpu(),
        )

    def predictions(self, feature_batch: architecture.FeatureBatch):
        return self.forward(feature_batch.stack())

    def generated(self, n, scale=0.9):
        encoding = torch.randn((n, 1 * 4 * 4), device=module_device(self)) * scale
        return architecture.PredictionBatch(
            images=self.decoder(encoding).cpu()
        )
