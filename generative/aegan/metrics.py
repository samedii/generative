import lantern


def gradient_metrics():
    return dict(
        image_loss=lantern.ReduceMetric(lambda _, image_loss, discriminator_loss: image_loss.item()),
        discriminator_loss=lantern.ReduceMetric(lambda _, image_loss, discriminator_loss: discriminator_loss.item()),
    )
