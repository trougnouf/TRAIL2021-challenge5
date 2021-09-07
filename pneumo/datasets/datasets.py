from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, CenterCrop
from utils import Augmenter
import os


def load(dataset_name, data_augmentation=False):
    if dataset_name == "pneumonia":
        return _from_image_folder(
            root="~/.cache/torch/mmf/data/pneumonia/train",
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
            transform=Compose([
                Resize(256),
                CenterCrop(224),
                Augmenter(ra=data_augmentation, prob=0.5),
                ToTensor(),
                #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
    else:
        raise Exception(f"Unknown dataset: {dataset_name}")


def _from_image_folder(root, transform):
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
    train = DataLoader(train_set, batch_size=32, shuffle=True)
    test = DataLoader(test_set, batch_size=32, shuffle=True)
    val = DataLoader(val_set, batch_size=32, shuffle=True)
    return train, test, val
