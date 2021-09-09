from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, CenterCrop
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform
from utils import Augmenter
from torch.utils.data import random_split, DataLoader
import os


def load(dataset_name, batch_size, data_augmentation=False):
    if dataset_name == "pneumonia":
        return _from_image_folder(
            root="~/.cache/torch/mmf/data/pneumonia",
            batch_size=batch_size,
            transform=Compose([
                Resize(256),
                CenterCrop(224),
                Augmenter(ra=False, prob=0.5),
                ToTensor(),
                #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
    elif dataset_name == "chestxray":
        return _from_image_folder(
            root="~/.cache/torch/mmf/data/chest_x_ray/single_label",
            batch_size=batch_size,
            transform=Compose([
                Resize(256),
                CenterCrop(224),
                Augmenter(ra=data_augmentation, prob=0.5),
                ToTensor(),
                #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
    elif dataset_name == "chexpert":
        return _load_chexpert_ssl(batch_size)
    else:
        raise Exception(f"Unknown dataset: {dataset_name}")


def _load_chexpert_ssl(batch_size):
    dataset = ImageFolder("~/.cache/torch/mmf/data/chexpert/train",
                          transform=SimCLRTrainDataTransform(input_height=224))
    num_samples = len(dataset)
    train_size = int(.8 * num_samples)
    val_size = int(.1 * num_samples)
    _test_size = int(.1 * num_samples)

    train, val, test = random_split(
        dataset,
        [train_size, val_size, num_samples - train_size - val_size]
    )
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_dl, test_dl, val_dl


def _from_image_folder(root, batch_size, transform):
    train_set = ImageFolder(
        root=os.path.join(root, "train"),
        transform=transform
    )
    test_set = ImageFolder(
        root=os.path.join(root, "test"),
        transform=transform
    )
    val_set = ImageFolder(
        root=os.path.join(root, "val"),
        transform=transform
    )
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train, test, val
