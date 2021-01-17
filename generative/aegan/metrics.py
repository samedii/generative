import lantern


def gradient_metrics():
    return dict(
        image_loss=lantern.ReduceMetric(lambda _, examples, reconstructed_image, image_loss, discriminator_loss: image_loss.item()),
        discriminator_loss=lantern.ReduceMetric(lambda _, examples, reconstructed_image, image_loss, discriminator_loss: discriminator_loss.item()),
        # mse=lantern.ReduceMetric(lambda _, examples, reconstructed_image, image_loss, discriminator_loss: reconstructed_image.mse(examples).item()),
    )
