from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lantern import FunctionalBase, Tensor

from generative import problem
from generative.tools import standardized


class StandardImage(FunctionalBase):
    data: Tensor

    @staticmethod
    def from_image(image: Image.Image):
        return StandardImage(data=standardized.encode(image))

    def representation(self, example=None):
        if example:
            return Image.fromarray(np.concatenate([
                np.array(example.image),
                np.array(standardized.decode(self.data)),
            ]))
        else:
            return standardized.decode(self.data)

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


class StandardImageBatch(FunctionalBase):
    data: Tensor

    @staticmethod
    def from_examples(examples: List[problem.Example]):
        return StandardImageBatch.from_images([example.image for example in examples])

    @staticmethod
    def from_images(images: List[Image.Image]):
        return StandardImageBatch(data=torch.stack([
            standardized.encode(image) for image in images
        ]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return StandardImage(
            data=self.data[index],
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def mse(self, standard_image):
        return F.mse_loss(
            self.data,
            standard_image.data,
        )

    def bce(self, standard_image):
        return F.binary_cross_entropy(
            (self.data + 1) / 2,
            (standard_image.data + 1) / 2,
        )

    def detach(self):
        return StandardImageBatch(data=self.data.detach())
