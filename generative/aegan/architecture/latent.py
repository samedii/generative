import numpy as np
import torch
import torch.nn.functional as F
from lantern import FunctionalBase, Tensor


class LatentBatch(FunctionalBase):
    loc: Tensor
    log_variance: Tensor

    @staticmethod
    def generated(batch_size):  # customize scale?
        shape = (batch_size, 1 * 4 * 4)
        return LatentBatch(
            loc=torch.zeros(shape),
            log_variance=torch.zeros(shape),
        )

    def encoding(self):
        return self.loc + torch.randn_like(self.loc) * torch.exp(0.5 * self.log_variance)

    # we can do better than this loss function if we want to have loc + scale
    def mse(self, latent):
        return F.mse_loss(
            self.encoding(),
            latent.encoding(),
        )

    def kl(self):
        kl = -0.5 * (1 + self.log_variance - self.loc ** 2 - self.log_variance.exp())
        return kl.mean()

    @property
    def scale(self):
        return (0.5 * self.log_variance).exp()

    def detach(self):
        return LatentBatch(
            loc=self.loc.detach(),
            log_variance=self.log_variance.detach(),
        )
