from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, CenterCrop
from utils import Augmenter, Normalizer

def load(dataset_name):
    # if dataset_name == "pneumonia":
    return _load_pneumonia()
    # else:
        # raise Exception(f"Unknown dataset: {dataset_name}")


def _load_pneumonia(image_size=(224, 224), batch_size=32):
    train_set = ImageFolder(
        root="/scratch/users/rvandeghen/xray/single_label/train",
        transform=Compose([
            Resize(256),
            CenterCrop(224),
            Augmenter(ra=False, prob=0.5),
            ToTensor(),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    test_set = ImageFolder(
        root="/scratch/users/rvandeghen/xray/single_label/test",
        transform=Compose([
            Resize(256),
            CenterCrop(224),
            Augmenter(ra=False, prob=0.5),
            ToTensor(),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    # val_set = ImageFolder(
    #     root="~/.cache/torch/mmf/data/pneumonia/val",
    #     transform=Compose([
    #         Resize(256),
    #         CenterCrop(224),
    #         Augmenter(ra=False, prob=0.5),
    #         ToTensor(),
    #         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    # )
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    # val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train, test#, val
