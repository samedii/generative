import argparse
import os
import json
from functools import partial
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.tensorboard
import lantern
from lantern import set_seeds, worker_init

from generative import datastream
from generative.aegan import architecture, metrics, log_reconstructed, log_generated


def train(config):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    set_seeds(config["seed"])

    models = dict(
        encoder=architecture.Encoder(),
        generator=architecture.Generator(),
        image_discriminator=architecture.ImageDiscriminator(),
        latent_discriminator=architecture.LatentDiscriminator(),
    )
    models = nn.ModuleDict({name: model.to(device) for name, model in models.items()})
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        for name, model in models.items()
    }

    if Path("model").exists():
        print("Loading model checkpoint")
        for name, model in models.items():
            model.load_state_dict(torch.load(f"model/{name}.pt"))

        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(torch.load(f"model/{name}.pt"))
            lantern.set_learning_rate(optimizer, config["learning_rate"])

    gradient_data_loader = datastream.GradientDatastream().data_loader(
        batch_size=config["batch_size"],
        num_workers=config["n_workers"],
        n_batches_per_epoch=config["n_batches_per_epoch"],
        worker_init_fn=partial(worker_init, config["seed"]),
        collate_fn=tuple,
    )

    evaluate_data_loaders = {
        f"evaluate_{name}": (
            datastream.take(128).data_loader(
                batch_size=config["eval_batch_size"],
                num_workers=config["n_workers"],
                collate_fn=tuple,
            )
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")
    gradient_metrics = lantern.Metrics(
        name="gradient",
        tensorboard_logger=tensorboard_logger,
        metrics=metrics.gradient_metrics(),
    )

    for epoch in lantern.Epochs(config["max_epochs"]):

        with lantern.module_train(models):
            for examples in lantern.ProgressBar(
                gradient_data_loader, metrics=gradient_metrics[["image_loss", "discriminator_loss"]]
            ):
                with torch.enable_grad():
                    real_image = architecture.StandardImageBatch.from_examples(examples)
                    real_latent = models["encoder"](real_image)
                    reconstructed_image = models["generator"](real_latent)

                    generated_latent = architecture.LatentBatch.generated(config["batch_size"])
                    generated_image = models["generator"](generated_latent)
                    reverse_latent = models["encoder"](reconstructed_image)

                    with torch.no_grad():
                        bce_latent_generated = models["latent_discriminator"](generated_latent).bce_real()
                        bce_image_reconstructed = models["image_discriminator"](reconstructed_image).bce_real()
                        bce_image_generated = models["image_discriminator"](generated_image).bce_real()

                    image_loss = (
                        reconstructed_image.bce(real_image)  # l1 instead? check cycle gan
                        + real_latent.mse(reverse_latent)  # l1 instead? check cycle gan
                        + bce_latent_generated
                        + bce_image_reconstructed
                        + bce_image_generated
                        + 0.1 * real_latent.kl()  # don't use this?
                    )
                    image_loss.backward()

                    bce_latent_real = models["latent_discriminator"](real_latent.detach()).bce_real()
                    bce_latent_generated = models["latent_discriminator"](generated_latent).bce_generated()

                    bce_image_real = models["image_discriminator"](real_image).bce_real()
                    bce_image_reconstructed = models["image_discriminator"](reconstructed_image.detach()).bce_generated()
                    bce_image_generated = models["image_discriminator"](generated_image.detach()).bce_generated()

                    discriminator_loss = (
                        bce_latent_real
                        + bce_latent_generated
                        + bce_image_real
                        + bce_image_reconstructed
                        + bce_image_generated
                    )
                    discriminator_loss.backward()

                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()

                gradient_metrics.update_(examples, reconstructed_image, image_loss, discriminator_loss).log_()
        gradient_metrics.print()
        log_reconstructed(tensorboard_logger, "gradient", epoch, examples, reconstructed_image)
        log_generated(tensorboard_logger, "gradient", epoch, generated_image)
        # tensorboard_logger.add_histogram("encoding", predictions.encoding, global_step=epoch)
        # tensorboard_logger.add_histogram("encoding_loc", predictions.loc, global_step=epoch)
        # tensorboard_logger.add_histogram("encoding_scale", predictions.scale, global_step=epoch)

        with lantern.module_eval(models):
            for name, data_loader in evaluate_data_loaders.items():
                for examples in tqdm(data_loader, desc=name, leave=False):
                    real_image = architecture.StandardImageBatch.from_examples(examples)
                    real_latent = models["encoder"](real_image)
                    reconstructed_image = models["generator"](real_latent)

                    generated_latent = architecture.LatentBatch.generated(config["batch_size"])
                    generated_image = models["generator"](generated_latent)

                    log_reconstructed(tensorboard_logger, name, epoch, examples, reconstructed_image)
                    log_generated(tensorboard_logger, name, epoch, generated_image)

        tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--n_batches_per_epoch", default=50, type=int)
    parser.add_argument("--n_batches_per_step", default=1, type=int)
    parser.add_argument("--n_workers", default=2, type=int)

    try:
        __IPYTHON__
        args = parser.parse_known_args()[0]
    except NameError:
        args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv("RUN_ID"),
    )

    Path("config.json").write_text(json.dumps(config))

    train(config)
