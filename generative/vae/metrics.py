import lantern


def gradient_metrics():
    return dict(
        loss=lantern.ReduceMetric(lambda _, examples, predictions, loss: loss.item()),
        bce=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.bce(examples).item()),
        mse=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.mse(examples).item()),
        kl=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.kl.item()),
        encoding_scale=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.scale.mean().item()),
    )


def evaluate_metrics():
    return dict(
        loss=lantern.MapMetric(lambda examples, predictions, loss: loss.item()),
        bce=lantern.MapMetric(lambda examples, predictions, loss: predictions.bce(examples).item()),
        mse=lantern.MapMetric(lambda examples, predictions, loss: predictions.mse(examples).item()),
        kl=lantern.MapMetric(lambda examples, predictions, loss: predictions.kl.detach().item()),
        encoding_scale=lantern.MapMetric(lambda examples, predictions, loss: predictions.scale.mean().item()),
    )
