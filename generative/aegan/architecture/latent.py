import numpy as np
import torch
import torch.nn.functional as F
from lantern import FunctionalBase, Tensor


class LatentBatch(FunctionalBase):
    encoding: Tensor

    @staticmethod
    def generated(batch_size, scale=1.0):
        return LatentBatch(
            encoding=torch.randn((batch_size, 1 * 4 * 4)) * scale,
        )

    def mse(self, latent):
        return F.mse_loss(
            self.encoding,
            latent.encoding,
        )

    def detach(self):
        return LatentBatch(encoding=self.encoding.detach())
