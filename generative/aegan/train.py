import argparse
import os
import json
from functools import partial
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.utils.tensorboard
from simple_pid import PID
from pydantic import BaseModel
import lantern
from lantern import set_seeds, worker_init

from generative import datastream
from generative.aegan import architecture, metrics, log_reconstructed, log_generated


class Controller(BaseModel):
    controllable: float
    target: float
    pid: PID

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, controllable, target):
        super().__init__(
            controllable=controllable,
            target=target,
            pid=Controller.new_pid(controllable, target)
        )

    @staticmethod
    def new_pid(controllable, target):
        pid = PID(
            0.01, 0.01, 0.0,
            setpoint=np.log10(target),
            auto_mode=False,
        )
        pid.set_auto_mode(True, last_output=np.log10(controllable))
        pid.output_limits = (None, -2)
        return pid

    def update_(self, current):
        self.controllable = 10 ** self.pid(np.log10(current), dt=1)
        return self.controllable

    def state_dict(self):
        return dict(controllable=self.controllable)

    def load_state_dict(self, state_dict):
        self.controllable = state_dict['controllable']
        self.pid = Controller.new_pid(self.controllable, self.target)


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

    image_controller = Controller(config["learning_rate"], 0.65)
    latent_controller = Controller(config["learning_rate"], 0.65)

    if Path("model").exists():
        print("Loading model checkpoint")
        models.load_state_dict(torch.load("model/models.pt"))

        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(torch.load(f"model/optimizer_{name}.pt"))
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
                gradient_data_loader, metrics=gradient_metrics[["autoencoder_loss", "discriminator_loss"]]
            ):
                with torch.enable_grad():
                    real_image = architecture.StandardImageBatch.from_examples(examples)
                    constructed_latent = models["encoder"](real_image)
                    reconstructed_image = models["generator"](constructed_latent)
                    reconstructed_latent = models["encoder"](reconstructed_image)

                    generated_latent = architecture.LatentBatch.generated(config["batch_size"])
                    generated_image = models["generator"](generated_latent)

                    with lantern.requires_nograd(models):
                        adversarial_latent_constructed = models["latent_discriminator"](constructed_latent).bce_real()
                        # adversarial_latent_reconstructed = models["latent_discriminator"](reconstructed_latent).bce_real()

                        adversarial_image_reconstructed = models["image_discriminator"](reconstructed_image).bce_real()
                        adversarial_image_generated = models["image_discriminator"](generated_image).bce_real()

                    autoencoder_loss = (
                        10 * reconstructed_image.l1(real_image)
                        + 0.1 * constructed_latent.mse(reconstructed_latent)
                        + 0.1 * adversarial_latent_constructed
                        # + 0.1 * adversarial_latent_reconstructed
                        + 0.1 * adversarial_image_reconstructed
                        + 0.01 * adversarial_image_generated
                    )
                    autoencoder_loss.backward()
                    nn.utils.clip_grad_norm_(models.parameters(), 1)

                    for name, optimizer in optimizers.items():
                        if "discriminator" not in name:
                            optimizer.step()
                            optimizer.zero_grad()

                    bce_latent_constructed = models["latent_discriminator"](constructed_latent.detach()).bce_constructed()
                    # bce_latent_reconstructed = models["latent_discriminator"](reconstructed_latent.detach()).bce_constructed()
                    bce_latent_generated = models["latent_discriminator"](generated_latent).bce_real()

                    bce_image_real = models["image_discriminator"](real_image).bce_real()
                    bce_image_reconstructed = models["image_discriminator"](reconstructed_image.detach()).bce_generated()
                    bce_image_generated = models["image_discriminator"](generated_image.detach()).bce_generated()

                    discriminator_loss = (
                        bce_latent_constructed
                        + bce_latent_generated
                        # + bce_latent_reconstructed
                        + bce_image_real
                        + 0.9 * bce_image_reconstructed
                        + 0.1 * bce_image_generated
                    )
                    discriminator_loss.backward()
                    nn.utils.clip_grad_norm_(models.parameters(), 1)

                    for name, optimizer in optimizers.items():
                        if "discriminator" in name:
                            optimizer.step()
                            optimizer.zero_grad()

                # nn.utils.clip_grad_norm_(models.parameters(), 1)

                # for optimizer in optimizers.values():
                #     optimizer.step()
                #     optimizer.zero_grad()

                gradient_metrics.update_(autoencoder_loss, discriminator_loss).log_()

                tensorboard_logger.add_scalar("gradient/reconstructed_image_l1", reconstructed_image.l1(real_image).item(), global_step=gradient_metrics.n_logs)
                tensorboard_logger.add_scalar("gradient/reconstructed_latent_mse", constructed_latent.mse(reconstructed_latent).item(), global_step=gradient_metrics.n_logs)

                real_image_probability = models["image_discriminator"](real_image).logits.sigmoid().mean().item()
                reconstructed_image_probability = models["image_discriminator"](reconstructed_image).logits.sigmoid().mean().item()
                tensorboard_logger.add_scalar("gradient/real_image_probability", real_image_probability, global_step=gradient_metrics.n_logs)
                tensorboard_logger.add_scalar("gradient/reconstructed_image_probability", reconstructed_image_probability, global_step=gradient_metrics.n_logs)
                tensorboard_logger.add_scalar("gradient/generated_image_probability", models["image_discriminator"](generated_image).logits.sigmoid().mean().item(), global_step=gradient_metrics.n_logs)

                constructed_latent_probability = models["latent_discriminator"](constructed_latent).logits.sigmoid().mean().item()
                generated_latent_probability = models["latent_discriminator"](generated_latent).logits.sigmoid().mean().item()
                tensorboard_logger.add_scalar("gradient/constructed_latent_probability", constructed_latent_probability, global_step=gradient_metrics.n_logs)
                tensorboard_logger.add_scalar("gradient/reconstructed_latent_probability", models["latent_discriminator"](reconstructed_latent).logits.sigmoid().mean().item(), global_step=gradient_metrics.n_logs)
                tensorboard_logger.add_scalar("gradient/generated_latent_probability", generated_latent_probability, global_step=gradient_metrics.n_logs)

                if gradient_metrics.n_logs >= 20:
                    image_controller.update_(real_image_probability)
                    latent_controller.update_(generated_latent_probability)

                    lantern.set_learning_rate(optimizers["image_discriminator"], image_controller.controllable)
                    lantern.set_learning_rate(optimizers["latent_discriminator"], latent_controller.controllable)

                tensorboard_logger.add_scalar("gradient/image_discriminator_learning_rate", image_controller.controllable, global_step=gradient_metrics.n_logs)
                tensorboard_logger.add_scalar("gradient/latent_discriminator_learning_rate", latent_controller.controllable, global_step=gradient_metrics.n_logs)

        gradient_metrics.print()

        log_reconstructed(tensorboard_logger, "gradient", epoch, examples, reconstructed_image)
        log_generated(tensorboard_logger, "gradient", epoch, generated_image)

        with lantern.module_eval(models):
            for name, data_loader in evaluate_data_loaders.items():
                for examples in tqdm(data_loader, desc=name, leave=False):
                    real_image = architecture.StandardImageBatch.from_examples(examples)
                    constructed_latent = models["encoder"](real_image)
                    reconstructed_image = models["generator"](constructed_latent)

                    generated_latent = architecture.LatentBatch.generated(config["batch_size"])
                    generated_image = models["generator"](generated_latent)

                    log_reconstructed(tensorboard_logger, name, epoch, examples, reconstructed_image)
                    log_generated(tensorboard_logger, name, epoch, generated_image)

        torch.save(models.state_dict(), "models.pt")
        for name, optimizer in optimizers.items():
            torch.save(optimizer.state_dict(), f"optimizer_{name}.pt")

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
