import numpy as np


def log_generated(logger, epoch, generated):
    logger.add_images(
        "generated",
        np.stack(
            [
                np.array(g.representation())[..., None] / 255
                for g in generated
            ]
        ),
        epoch,
        dataformats="NHWC",
    )
