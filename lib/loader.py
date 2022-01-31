import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from lib.pipeline import Sample, get_transforms
from typing import List


class FER2013(torch.utils.data.Dataset):
    def __init__(self, args, mode=None, transform: transforms = None):
        self.args = args
        self._mode = mode  # "train" or "test"
        self._data_pd, self._data = self._load_data()
        self._transform = transform

    def _load_data(self):
        if self._mode == "train":
            print("Load training data")
            _data = pd.read_csv(self.args.FER2013PTR)
        elif self._mode == "test":
            print("Load testing data")
            _data = pd.read_csv(self.args.FER2013PTE)
        elif self._mode == "valid":
            print("Load valid data")
            _data = pd.read_csv(self.args.FER2013PVA)
        else:  # raise an error
            _data = None

        _imgs = [[int(y) for y in x.split()] for x in _data['pixels']]
        _labels = _data['emotion'].tolist()

        return _data, list(zip(_imgs, _labels))

    def __len__(self):
        return len(self._data_pd)

    def __getitem__(self, idx: int) -> Sample:
        image = Image.fromarray(np.uint8(np.reshape(self._data[idx][0], (48, 48))))
        image = transforms.Grayscale(num_output_channels=3)(image)
        sample = Sample({
            "image": image,
            "label": self._data[idx][1]})

        if self._transform:
            sample = self._transform(sample)

        return sample


def FER2013_dataloader(args, transforms_new_size):
    """Instantiate the FER2013 class and return a PyTorch Dataloader."""
    transformer = get_transforms(transforms_new_size)

    trainset = FER2013(args=args, mode="train", transform=transformer)
    testset = FER2013(args=args, mode="test", transform=transformer)
    valset = FER2013(args=args, mode="valid", transform=transformer)

    print(f"created trainset of size: {len(trainset)}")
    print(f"created testset of size: {len(testset)}")
    print(f"created valset of size: {len(valset)}")

    traindl = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    testdl = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    valdl = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    return traindl, testdl, valdl


# # # # # # # # # # # # # # # # # # # # # # # # #
#
#               JAFFE Dataloader
#
# # # # # # # # # # # # # # # # # # # # # # # # #

class JAFFE(torch.utils.data.Dataset):
    """Implementation of the JAFFE dataset by subclassing PyTorch's dataset class."""

    def __init__(self, path_to_dataset: Path, mode=None, split_dataset: float = None, transform: transforms = None):
        """
        path_to_dataset:
        transform:
        """
        self._mode = mode
        self._split_dataset = split_dataset
        self._path_dataset: Path = path_to_dataset
        self._dataset = [f for f in self._path_dataset.iterdir() if f.is_file()]
        # get train, test and validation split of image paths
        self._path_images_train, self._path_images_test, self._path_images_val = self._split(self._dataset, 0.5,
                                                                                             self._split_dataset)
        print(f"inside init function: train {len(self._path_images_train)}, test {len(self._path_images_test)}")
        self._path_images = None
        self._transform = transform
        self._classes = {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE': 6}

    def get_split(self, mode: str):
        """ Return the training, testing or validation JAFFE dataset
        :mode: "train", "test", "val"
        return: JAFFE dataset
        """
        print(f"Returning {mode}-set.")
        self._mode = mode
        if mode == "train":
            self._path_images = self._path_images_train
            new_instance = deepcopy(self)
        elif mode == "test":
            self._path_images = self._path_images_test
            new_instance = deepcopy(self)
        elif mode == "val":
            self._path_images = self._path_images_val
            new_instance = deepcopy(self)
        return new_instance

    def _split(self, dataset: List, testval=0.5, split_dataset=0.8):
        """Splits the image paths into train, test image paths
        split_dataset: Has to be a float in range [0.0, 1.0]
        test_val: splits the test set into test/val of same size"""
        lendata = len(dataset)
        np.random.shuffle(dataset)

        train_amount = split_dataset

        bound = int(train_amount * lendata)
        train, test = dataset[:bound], dataset[bound:]
        len_testset = len(test)
        newtest, val = test[:int(testval * len_testset)], test[int(testval * len_testset):]

        return train, newtest, val

    def show_sample(self, idx: int):
        """Given a index idx this function shows the corresponding image sample"""
        with Image.open(self._path_images[idx]) as img:
            display(img)

    def _load_image(self, path: Path):
        """Load image using path and returns this image as np.Array"""
        image = PIL.Image.open(path)
        image = transforms.Grayscale(num_output_channels=3)(image)
        return image

    def _get_label(self, filepath: Path) -> str:
        """Extracts label from the filename of an JAFFE image sample"""
        return filepath.stem.split(".")[1][:2]

    def __len__(self):
        return len(self._path_images)

    def __getitem__(self, idx: int) -> Sample:
        tmp_path = self._path_images[idx]

        sample = Sample({
            "image": self._load_image(tmp_path),
            "label": self._classes[self._get_label(tmp_path)]})

        if self._transform:
            sample = self._transform(sample)

        return sample

    def _download(self):
        pass


def JAFFE_dataloader(path_to_dataset: Path, split_dataset, transforms_new_size):
    """Instantiate the jaffe class and return a PyTorch Dataloader."""
    transformer = get_transforms(transforms_new_size)
    dataset = JAFFE(path_to_dataset=path_to_dataset,
                    mode=None,
                    split_dataset=0.8,
                    transform=transformer)

    trainset = dataset.get_split("train")
    testset = dataset.get_split("test")
    valset = dataset.get_split("val")

    print(f"created trainset of size: {len(trainset)}")
    print(f"created testset of size: {len(testset)}")
    print(f"created valset of size: {len(valset)}")

    traindl = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    testdl = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    valdl = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    return traindl, testdl, valdl


def get_loaders(args):
    # traindl, testdl, valdl = JAFFE_dataloader(path_to_dataset=dataset_path, split_dataset = 0.8, transforms_new_size=[224, 224])

    if args.DATASET == "FER":
        traindl, testdl, valdl = FER2013_dataloader(args=args, transforms_new_size=[224, 224])
    #elif args.DATASET == "JAFFE":
    #    pass

    return traindl, testdl, valdl