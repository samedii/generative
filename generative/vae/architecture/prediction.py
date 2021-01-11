from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from lantern import FunctionalBase, Tensor


class Prediction(FunctionalBase):
    image: Tensor

    def representation(self, example=None):
        prediction = np.clip(self.image.numpy(), 0, 1) * 255
        if example:
            return Image.fromarray(np.concatenate([
                np.array(example.image),
                prediction,
            ]))
        else:
            return Image.fromarray(prediction)

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


class PredictionBatch(FunctionalBase):
    images: Tensor.shape(-1, 32, 32)
    loc: Optional[Tensor]
    log_variance: Optional[Tensor]
    encoding: Optional[Tensor]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return Prediction(
            image=self.images[index],
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def stack_images(self, examples):
        return torch.as_tensor(
            np.stack([example.image for example in examples]),
            device=self.images.device,
            dtype=torch.float32,
        ) / 255

    def loss(self, examples):
        return (
            self.bce(examples)
            + 0.1 * self.kl
        )

    def mse(self, examples):
        return F.mse_loss(
            self.images,
            self.stack_images(examples),
        )

    def bce(self, examples):
        return F.binary_cross_entropy(
            self.images,
            self.stack_images(examples),
        )

    @property
    def kl(self):
        #  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = -0.5 * (1 + self.log_variance - self.loc ** 2 - self.log_variance.exp())
        # return loss.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        return kl.mean()

    @property
    def scale(self):
        return (0.5 * self.log_variance).exp()
