from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from lantern import FunctionalBase, Tensor


class Prediction(FunctionalBase):
    image: Tensor

    def representation(self, example=None):
        if example:
            image = example.image
        else:
            image = Image.new("L", (32, 32))

        return np.concatenate([
            np.array(image),
            np.clip(self.image.numpy(), 0, 1) * 255,
        ])

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


class PredictionBatch(FunctionalBase):
    images: Tensor.shape(-1, 32, 32)

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
        return self.mse(examples)

    def mse(self, examples):
        return F.mse_loss(
            self.images,
            self.stack_images(examples),
        )
