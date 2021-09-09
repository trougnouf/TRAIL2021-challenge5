"""Functions common to train/test of emotions recognizer."""

import os
import sys
import torch
import torchvision
from pl_bolts.models.self_supervised import SwAV, SimCLR, SimSiam


DS_ROOT = os.path.join(
    "..", "datasets"
)  # must be placed before local imports (circular import)

sys.path.append("..")
from emotions.tools import (
    prep_raf_dataset,
    prep_kdef_dataset,
)

MODELS_DPATH = os.path.join("..", "models", "emotions")
NUM_CLASSES = 7

DS_BUILDERS = {
    "raf": prep_raf_dataset.make_raf_ImageFolder_struct,
    "kdef": prep_kdef_dataset.make_KDEF_ImageFolder_struct,
}


def add_common_arguments(parser):
    """Add parser arguments common to train and test."""
    parser.add_argument(
        "--config",
        is_config_file=True,
        dest="config",
        required=False,
        help="config in yaml format",
    )
    parser.add_argument(
        "--pretrain_url",
        help="URL to pre-trained model (default downloads SwAV-ImageNet)",
    )
    parser.add_argument("--pretrain_fpath", help="File path to trained model")
    parser.add_argument(
        "--test_ds_names",
        nargs="+",
        help="Dataset names (directories in ../datasets/emotions/test)",
    )
    parser.add_argument(
        "--device",
        help="Device number used if cuda is detected",
    )
    parser.add_argument(
        "--model_pretrain_method",
        help="Which model to train, try Swav, SimCLR or SimSiam",
    )


def init_model(model_pretrain_method, pretrain_url, device):
    if model_pretrain_method == "Swav":
        model = SwAV.load_from_checkpoint(pretrain_url, strict=True).model
        model.prototypes = torch.nn.Linear(128, NUM_CLASSES)
    elif model_pretrain_method == "SimCLR":
        try:
            model = SimCLR.load_from_checkpoint(pretrain_url, strict=False).encoder
        except FileNotFoundError as e:
            print(
                f'error: unable to load model. hint: have you run "bash tools/download_pretrained_models.sh"? ({e})'
            )
            exit(-1)
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=NUM_CLASSES, bias=True
        )
    elif model_pretrain_method == "SimSiam":
        try:
            model = SimSiam.load_from_checkpoint(
                pretrain_url, strict=True
            ).online_network
        except FileNotFoundError as e:
            print(
                f'error: unable to load model. hint: have you run "bash tools/download_pretrained_models.sh"? ({e})'
            )
            exit(-1)
        model = model.encoder
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=NUM_CLASSES, bias=True
        )
    elif model_pretrain_method == "torchvision":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=NUM_CLASSES, bias=True
        )
    else:
        raise NotImplementedError(model_pretrain_method)
    model = model.to(device)
    return model


def fw(model, input, model_pretrain_method):
    """Call and index forward according to model type."""
    output = model(input)
    if model_pretrain_method == "torchvision":
        return output
    elif model_pretrain_method == "Swav":
        return output[1]
    else:
        return output[0]


def get_dataloaders(
    train_ds_names: list,
    test_ds_names: list,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    ds_root: str = DS_ROOT,
):
    """
    Get train and test data loaders.

    Based on directories in ../datasets/[train or test].

    train_ds_names can be an empty list; returned train_loader will be None.
    """
    train_sets = []
    test_sets = []
    for ds_name in train_ds_names:
        if not os.path.isdir(os.path.join(DS_ROOT, "train", ds_name)):
            print(f"Dataset not found; building with {DS_BUILDERS[ds_name]}")
            DS_BUILDERS[ds_name]()

        train_set = torchvision.datasets.ImageFolder(
            root=os.path.join(ds_root, "train", ds_name),
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(image_size),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        train_sets.append(train_set)
    for ds_name in test_ds_names:
        test_set = torchvision.datasets.ImageFolder(
            root=os.path.join(ds_root, "test", ds_name),
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(image_size),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        test_sets.append(test_set)
    if len(train_sets) > 0:
        train_combisets = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(
            train_combisets, batch_size=batch_size, shuffle=True
        )
    else:
        train_loader = None
    test_combisets = torch.utils.data.ConcatDataset(test_sets)
    test_loader = torch.utils.data.DataLoader(
        test_combisets, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
