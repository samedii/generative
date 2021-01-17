from PIL import Image
import numpy as np
import torch

from lantern import Tensor


def encode(image: Image.Image) -> Tensor:
    return (
        torch.as_tensor(np.array(image, dtype=np.float32))[None] / 255 * 2 - 1
    )


def decode(standard_image: Tensor) -> Image.Image:
    return Image.fromarray(np.uint8((standard_image[0].clamp(-1, 1).numpy() + 1) / 2 * 255))
