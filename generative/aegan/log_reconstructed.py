import numpy as np


def log_reconstructed(logger, name, epoch, examples, reconstructed_images):
    n_examples = min(5, len(reconstructed_images))
    indices = np.random.choice(
        len(reconstructed_images),
        n_examples,
        replace=False,
    )
    logger.add_images(
        f"{name}/reconstructed",
        np.stack(
            [
                np.array(
                    reconstructed_images[index].representation(examples[index])
                )[..., None] / 255
                for index in indices
            ]
        ),
        epoch,
        dataformats="NHWC",
    )
