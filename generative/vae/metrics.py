import lantern


def gradient_metrics():
    return dict(
        loss=lantern.ReduceMetric(lambda _, examples, predictions, loss: loss),
        bce=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.bce(examples)),
        mse=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.mse(examples)),
        kl=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.kl.detach()),
        encoding_scale=lantern.ReduceMetric(lambda _, examples, predictions, loss: predictions.scale.detach().mean()),
    )


def evaluate_metrics():
    return dict(
        loss=lantern.MapMetric(lambda examples, predictions, loss: loss),
        bce=lantern.MapMetric(lambda examples, predictions, loss: predictions.bce(examples)),
        mse=lantern.MapMetric(lambda examples, predictions, loss: predictions.mse(examples)),
        kl=lantern.MapMetric(lambda examples, predictions, loss: predictions.kl.detach()),
        encoding_scale=lantern.MapMetric(lambda examples, predictions, loss: predictions.scale.detach().mean()),
    )
