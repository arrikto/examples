from typing import Tuple
from argparse import ArgumentParser, Namespace

import torch
import torch.utils.data as data
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, Food101


def _split_data(dataset: data.Dataset,
                train_split: float = 0.8,
                seed: int = 42) -> Tuple[data.Dataset, data.Dataset]:
    """Split a dataset into two subsets.

    Args:
        dataset (data.Dataset): Dataset to split.
        split (float): Percentage of data to use for the train subset.
        seed (int): Seed for the random number generator.
    
    Returns:
        Tuple[data.Dataset, data.Dataset]: The two subsets.
    """
    # use a percentage of training data for validation
    train_set_size = int(len(dataset) * train_split)
    valid_set_size = len(dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(seed)
    train_set, valid_set = data.random_split(
        dataset, [train_set_size, valid_set_size], generator=seed)

    return train_set, valid_set


def add_data_args(parser: ArgumentParser):
    """Add data arguments to the parser.
    
    Args:
        parser (ArgumentParser): The parser to add the arguments to.
    
    Returns:
        ArgumentParser: The parser with the added arguments.
    """
    parser = ArgumentParser(parents=[parser], add_help=False)

    parser.add_argument("--download_path", type=str, default=".data",
                        help="The path to download the dataset to.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch size to use.")
    parser.add_argument("--image_size", type=int, default=32,
                        help="The size of the images to use.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="The number of workers to use in"
                             " the DataLoader.")
    parser.add_argument("--drop_last", action="store_true", default=False,
                        help="Whether to drop the last batch if it is"
                             " smaller than the batch size.")
    parser.add_argument("--pin_memory", action="store_true", default=False,
                        help="Whether to automatically put fetched data"
                             " in pinned memory, to enable faster data"
                             " transfer to CUDA-enabled GPUs.")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="The percentage of the training set to use"
                             " for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed to use for the random split of"
                             " the training set.")

    return parser


class MNISTDataset(pl.LightningDataModule):
    """Define a LightningDataModule for the MNIST dataset.

    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ is a dataset of handwritten
    digits. It consists of 60,000 training images and 10,000 test images. Each
    image is a 28x28 grayscale image, associated with a label from 0-9
    representing the digit drawn in the image.

    Args:
        args (Namespace): Arguments.
    
    Attributes:
        download_path (str): Path to download the dataset.
        batch_size (int): Batch size.
        image_size (int): Size of the images.
        num_workers (int): Number of workers.
        drop_last (bool): Drop the last batch if it is smaller than the batch
            size.
        pin_memory (bool): Pin memory on host.
        train_split (float): Percentage of training data to use for validation.
        seed (int): Seed for the random number generator.
    """

    def __init__(self, args: Namespace):
        super().__init__()

        self.download_path = args.download_path
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.num_workers
        self.drop_last = args.drop_last
        self.pin_memory = args.pin_memory
        self.train_split = args.train_split
        self.seed = args.seed

        self._load_data()

    def _load_data(self):
        """Split the training set into a training and validation set."""

        transform = transforms.Compose([
            # resize to 32x32 so it works with the UNet model architecture
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()])
        
        dataset = MNIST(
            root=self.download_path, transform=transform, download=True)

        self.train_set, self.valid_set = _split_data(
            dataset, self.train_split, self.seed)

    def train_dataloader(self):
        """Return a DataLoader for the training set."""
        train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        """Return a DataLoader for the validation set."""
        val_dataloader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory)

        return val_dataloader

    def test_dataloader(self):
        """Return a DataLoader for the test set."""
        transform = transforms.Compose([transforms.ToTensor()])
        
        test_dataset = MNIST(self.download_path, train=False,
                             transform=transform, download=True)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,)

        return test_dataloader


class Food101Dataset(pl.LightningDataModule):
    """Define a LightningDataModule for the Food101 dataset.

    `Food101 <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_ is
    a dataset of 101 food categories, with 101,000 images.

    Args:
        args (Namespace): Arguments.
    
    Attributes:
        download_path (str): Path to download the dataset.
        batch_size (int): Batch size.
        image_size (int): Size of the images.
        num_workers (int): Number of workers.
        drop_last (bool): Drop the last batch if it is smaller than the batch
            size.
        pin_memory (bool): Pin memory on host.
        train_split (float): Percentage of training data to use for validation.
        seed (int): Seed for the random number generator.
    """

    def __init__(self, args: Namespace):
        super().__init__()

        self.download_path = args.download_path
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.num_workers
        self.drop_last = args.drop_last
        self.pin_memory = args.pin_memory
        self.train_split = args.train_split
        self.seed = args.seed

        self._load_data()

    def _load_data(self):
        """Split the training set into a training and validation set."""

        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
        dataset = Food101(
            root=self.download_path, transform=transform, download=True)

        self.train_set, self.valid_set = _split_data(
            dataset, self.train_split, self.seed)

    def train_dataloader(self):
        """Return a DataLoader for the training set."""
        train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        """Return a DataLoader for the validation set."""
        val_dataloader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory)

        return val_dataloader

    def test_dataloader(self):
        """Return a DataLoader for the test set."""
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
        
        test_dataset = MNIST(self.download_path, split="test",
                             transform=transform, download=True)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,)

        return test_dataloader
