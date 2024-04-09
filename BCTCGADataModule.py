import os
from typing import Optional

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets import MNIST

DEFAULT_DATA_DIR = "./data"
DEFAULT_NUM_WORKERS = 32

class BCTCGADataModule(LightningDataModule):
    def __init__(
        self,
        augment: dict = None,
        batch_size: int = 32,
        data_dir: str = DEFAULT_DATA_DIR,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_sampler: Sampler = None,
        tune_on_val: float = 0,
        seed: int = None,
    ):
        super().__init__()
        self._has_setup_attack = False

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.num_classes = 10

        self.batch_sampler = batch_sampler
        self.tune_on_val = tune_on_val
        self.multi_class = False
        self.seed = seed

        bctcga_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                               (0.2023, 0.1994, 0.2010))

        self._train_transforms = [transforms.ToTensor(), bctcga_normalize]
        if augment["hflip"]:
            self._train_transforms.insert(
                0, transforms.RandomHorizontalFlip(p=0.5))
        if augment["color_jitter"] is not None:
            self._train_transforms.insert(
                0,
                transforms.ColorJitter(
                    brightness=augment["color_jitter"][0],
                    contrast=augment["color_jitter"][1],
                    saturation=augment["color_jitter"][2],
                    hue=augment["color_jitter"][3],
                ),
            )
        if augment["rotation"] > 0:
            self._train_transforms.insert(
                0, transforms.RandomRotation(augment["rotation"]))
        if augment["crop"]:
            self._train_transforms.insert(0,
                                          transforms.RandomCrop(32, padding=4))

        print(self._train_transforms)

        self._test_transforms = [transforms.ToTensor(), cifar_normalize]

        self.prepare_data()

    def prepare_data(self):
        #data loaded into ./data

    def setup(self, stage: Optional[str] = None):
        """Initialize the dataset based on the stage option ('fit', 'test' or 'attack'):
        - if stage is 'fit', set up the training and validation dataset;
        - if stage is 'test', set up the testing dataset;
        - if stage is 'attack', set up the attack dataset (a subset of training images)

        Args:
            stage (Optional[str], optional): stage option. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            if self.tune_on_val:
                self.val_set = CIFAR10(
                    self.data_dir,
                    train=True,
                    transform=transforms.Compose(self._test_transforms),
                )
                train_indices, val_indices = train_val_split(
                    len(self.train_set), self.tune_on_val)
                self.train_set = Subset(self.train_set, train_indices)
                self.val_set = Subset(self.val_set, val_indices)
            else:
                self.val_set = CIFAR10(
                    self.data_dir,
                    train=False,
                    transform=transforms.Compose(self._test_transforms),
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CIFAR10(
                self.data_dir,
                train=False,
                transform=transforms.Compose(self._test_transforms),
            )

        if stage == "attack":
            ori_train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, seed=self.seed)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_mini":
            ori_train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=2)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_large":
            ori_train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=500)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))

    def train_dataloader(self):
        if self.batch_sampler is None:
            return DataLoader(self.train_set,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers)
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
