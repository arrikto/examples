import time

from argparse import ArgumentParser

import torch
import torch.distributed as dist
import numpy as np

from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import Callback
from torchmetrics.image.fid import FrechetInceptionDistance

from utils import tensor_to_image


class DDPMCallback(Callback):
    """Add noise to the images during training.
    
    Args:
        args (Namespace): The arguments to construct the callback.
    
    Attributes:
        scheduler (DDPMScheduler): The noise scheduler.
    """
    def __init__(self, args):
        self.scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule)

    def _sample(self, model, size):
        # Sample noise
        torch.manual_seed(0)
        noise = torch.randn(size, device=model.device)

        progress = []

        for i, t in enumerate(self.scheduler.timesteps):
            with torch.no_grad():
                # get the noise prediction
                noise_pred = model(noise, t)

            # compute the previous noisy sample x_t -> x_t-1
            noise = self.scheduler.step(noise_pred, t, noise).prev_sample

            if (i+1) % 62.5 == 0:
                progress.append(tensor_to_image(noise))

        return noise, progress

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        images, _ = batch
        # Sample noise
        noise = torch.randn(images.shape).to(images.device)
        # Sample a random timestep for each image
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (batch_size,),
            device=images.device).long()
        # Add noise to the images
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)

        # Construct a new example batch. The features are the noisy images
        # and the timestep and the label is the noise we added.
        batch[0] = (noisy_images, timesteps)
        batch[1] = noise

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx,
                                  dataloader_idx):
        return self.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx,
                            dataloader_idx):
        return self.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        # Sample a batch of images
        batch = next(iter(trainer.train_dataloader))
        _, C, H, W = batch[0].shape

        noise, progress = self._sample(pl_module, (1, C, H, W))
        image = tensor_to_image(noise)

        trainer.logger.experiment.add_images(
                "denoising process", torch.tensor(np.array(progress)),
                global_step=trainer.global_step)

        trainer.logger.experiment.add_image(
            "sample", image, global_step=trainer.global_step)

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     # Sample a batch of images
    #     batch = next(iter(trainer.val_dataloaders[0]))
    #     B, C, H, W = batch[0].shape

    #     noise, _ = self._sample(pl_module, (B, C, H, W))
    #     noise = noise.detach().cpu()

    #     image = batch[0].to(noise.device)

    #     fid = FrechetInceptionDistance(
    #         feature=64, normalize=True).to(noise.device)

    #     fid.update(image.repeat(1,3,1,1), real=True)
    #     fid.update(noise.repeat(1,3,1,1), real=False)
    #     fid_score = fid.compute()

    #     trainer.logger.log_metrics(
    #         {"FID": fid_score}, step=trainer.global_step)

    @staticmethod
    def add_callback_args(parser: ArgumentParser) -> ArgumentParser:
        """Add callback arguments to the parser.
        
        Args:
            parser (ArgumentParser): The parser to add the arguments to.
        
        Returns:
            ArgumentParser: The parser with the added arguments.
        """
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument("--ddpm_num_steps", type=int, default=1000,
                            help="Number of training steps for DDPM.")
        parser.add_argument("--ddpm_beta_schedule", type=str, default="linear",
                            help="The beta schedule for DDPM.")

        return parser


class MetricsCallback(Callback):
    def __init__(self, args):
        self.args = args
    def on_train_start(self, trainer, pl_module):
        self.t0_train = time.time()
        self.throughput = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.t0_epoch = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        t1 = time.time()

        # Calculate epoch elapsed time
        dt = t1 - self.t0_epoch
        
        # Calculate the number of examples
        n_examples = len(trainer.train_dataloader.dataset)

        # Calculate the throughput
        world_size = dist.get_world_size() if dist.is_available() else 1
        images_per_sec = (n_examples / dt) * world_size

        self.throughput.append(images_per_sec)

        pl_module.log("images_per_sec", images_per_sec)
        pl_module.log("total_epoch_time", dt)

    def on_train_end(self, trainer, pl_module):
        throughput = np.mean(self.throughput)

        params = {
            "batch_size": self.args.batch_size,
            "num_workers": self.args.num_workers,
            "drop_last": self.args.drop_last,
            "pin_memory": self.args.pin_memory,
            "image_size": self.args.image_size,
            "precision": trainer.precision,
        }
        metrics = {"throughput": throughput}
        trainer.logger.log_hyperparams(params=params, metrics=metrics)
