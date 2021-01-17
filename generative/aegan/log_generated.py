import numpy as np


def log_generated(logger, name, epoch, generated):
    logger.add_images(
        f"{name}/generated",
        np.stack(
            [
                np.array(g.representation())[..., None] / 255
                for g in generated
            ]
        ),
        epoch,
        dataformats="NHWC",
    )
