import sys
import logging

from argparse import ArgumentParser

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

from model import UNet
from callbacks import DDPMCallback, MetricsCallback
from data import MNISTDataset, Food101Dataset, add_data_args
from utils import setup_distributed_training, setup_logger, setup_profiler


# Set the logging level to INFO for all loggers
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.INFO)


def main():
    pl.seed_everything(42)

    parser = ArgumentParser(description="PyTorch Food101 Example")
    # Add Food101 training related arguments
    parser.add_argument("--experiment_name", type=str, default="ddpm",
                        help="Name of the experiment")
    parser.add_argument("--version", type=str, default="0",
                        help="The version of the experiment")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="The dataset to use")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="Disables CUDA training.")
    parser.add_argument("--cpu_offload", action="store_true", default=False,
                        help="Enables CPU offloading for FSDP strategy.")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Uses the PyTorch Profiler to track computation,"
                             " exported as a Chrome-style trace.")
    parser.add_argument("--logdir", type=str, default="/logs",
                        help="Path to save the logs.")

    # Add training related arguments
    parser = pl.Trainer.add_argparse_args(parser)
    # Add dataset related arguments
    parser = add_data_args(parser)
    # Add callback related arguments
    parser = DDPMCallback.add_callback_args(parser)
    # Add model related arguments
    parser = UNet.add_model_specific_args(parser)
    # Parse arguments
    args = parser.parse_args()

    # Instantiate the model
    model = UNet(args)
    
    # Instantiate the dataset
    if args.dataset == "mnist":
        data = MNISTDataset(args)
    if args.dataset == "food101":
        data = Food101Dataset(args)

    # Define the logger
    logger, experiment_dir = setup_logger(args)

    # Define the distributed training configuration
    distributed_config = setup_distributed_training(args)

    # Define the checkpoint callback
    callbacks = [
        DDPMCallback(args),
        MetricsCallback(args),
        ModelCheckpoint(monitor='val_loss', mode='min', dirpath=experiment_dir)
    ]

    # Define and fit the trainer
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks, **distributed_config)

    # Define the profiler
    profiler = setup_profiler(args, experiment_dir)
    trainer.profiler = profiler

    trainer.fit(model, data)


if __name__ == "__main__":
    sys.exit(main())
