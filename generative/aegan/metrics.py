import lantern


def gradient_metrics():
    return dict(
        autoencoder_loss=lantern.ReduceMetric(lambda _, autoencoder_loss, discriminator_loss: autoencoder_loss.item()),
        discriminator_loss=lantern.ReduceMetric(lambda _, autoencoder_loss, discriminator_loss: discriminator_loss.item()),
    )
