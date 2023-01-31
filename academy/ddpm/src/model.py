from typing import Tuple
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F

from diffusers import UNet2DModel
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR


class UNet(LightningModule):
    """Define a UNet model architecture for the denoising process.

    Args:
        in_channels (int): Number of input channels. Defaults to 1.
        out_channels (int): Number of output channels. Defaults to 1.
        block_out_channels (Tuple): Number of output channels for each block.
            Defaults to (32, 64, 128, 128).

    Attributes:
        model (UNet2DModel): The UNet model.
        lr (float): The learning rate.
        max_lr (float): The maximum learning rate for 1 cycle training.
        total_steps (int): The total number of training steps.
    """
    def __init__(self, args: Namespace):
        super(UNet, self).__init__()
        # Construct the UNet model
        blocks = args.block_out_channels
        blocks = [int(b) for b in args.block_out_channels[0].split(" ")]
        self.model = UNet2DModel(
            in_channels=args.in_channels, out_channels=args.out_channels,
            block_out_channels=tuple(blocks))

        # Create a dummy input to get the computational graph
        image = torch.zeros(
            (1, args.in_channels, args.image_size, args.image_size))
        timestep = torch.randint(
            0, 1000, (1,)).long()
        self.example_input_array = image, timestep

        self.lr = args.lr
        self.max_lr = args.max_lr
        self.total_steps = args.max_steps

    def forward(self, sample, timesteps):
        return self.model(sample, timesteps).sample
    
    def training_step(self, batch, batch_idx):
        (sample, timesteps), noise = batch
        preds = self(sample, timesteps)
        loss = F.mse_loss(preds, noise)

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        test_loss = self.training_step(batch, batch_idx)

        self.log("test_loss", test_loss)
        return {"test_loss": test_loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.
        
        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer =  torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.max_lr is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": OneCycleLR(optimizer, max_lr=self.max_lr,
                                        total_steps=self.total_steps),
                "monitor": "val_loss",
                "frequency": 5}}


    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments to the parser.
        
        Args:
            parent_parser (ArgumentParser): The parser to add the arguments to.
        
        Returns:
            ArgumentParser: The parser with the added arguments.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--in_channels", type=int, default=1,
                            help="Number of input channels.")
        parser.add_argument("--out_channels", type=int, default=1,
                            help="Number of output channels.")
        parser.add_argument("--block_out_channels", type=str, nargs='+',
                            default=(32, 64, 128, 128),
                            help="Number of output channels for each block.")
        parser.add_argument("--lr", type=float, default=1e-3,
                            help="The learning rate of the optimizer.")
        parser.add_argument("--max_lr", type=float, default=None,
                            help="The maximum learning rate of the optimizer"
                                 " for 1 cycle training.")
        return parser
