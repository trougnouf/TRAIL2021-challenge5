"""Train loop for RAF model with PyTorch Lightning."""

import os
import pathlib
import torch
import numpy as np
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.tensorboard import SummaryWriter
from pl_bolts.models.self_supervised import SwAV
from datetime import datetime
import random
import sys
import datetime

sys.path.append("..")
from emotions.tools import prep_raf_dataset

random.seed(42)

MODELS_DPATH = os.path.join("..", "models", "emotions")
RAF_DATA_DPATH = os.path.join("..", "datasets", "raf_basic")
PATIENCE=10
LR_DECAY=0.5
N_EPOCHS=200

if not os.path.isdir(RAF_DATA_DPATH):
    print(
        "RAF_DATA_DPATH ({RAF_DATA_DPATH}) not found; building with tools/prep_raf_dataset.py"
    )
    prep_raf_dataset.make_raf_ImageFolder_struct()


def get_raf_dataloaders(
    image_size=(224, 224), batch_size=32, data_dpath=RAF_DATA_DPATH
):
    """
    Load a RAF dataset from data_dpath containing train_* and test_* images.

    Return train, test, val (10% of train) dataloaders.
    """
    train_set = ImageFolder(
        root=os.path.join(data_dpath, "train"),
        transform=Compose([Resize(image_size), ToTensor()]),
    )
    test_set = ImageFolder(
        root=os.path.join(data_dpath, "test"),
        transform=Compose([Resize(image_size), ToTensor()]),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def train(
    dataset,
    model,
    loss_function,
    optimizer,
    device,
    n_epochs,
    val_dataset,
    save_dpath,
    patience,
    lr_decay,
):
    """Training loop."""
    print("Started training")
    writer = SummaryWriter(os.path.join(save_dpath, "tensorboard"))
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # simple lr decay every patience epochs
        if epoch % patience == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay
                print('train: reduced lr to param_group["lr"]')
        running_loss = 0.0
        evaluation = []
        for i, data in enumerate(dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)[1]
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            evaluation.append((outputs.argmax(1) == labels).cpu().sum().item())

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                running_loss = running_loss / 10
                print(
                    "[%d, %5d/%d] loss: %.3f"
                    % (epoch + 1, i + 1, len(dataset), running_loss)
                )
                writer.add_scalar(
                    "Running Loss/train", running_loss, epoch * len(dataset) + i
                )
                writer.add_scalar(
                    "Accuracy/train", np.mean(evaluation) / 32, epoch * len(dataset) + i
                )
                evaluation = []
                running_loss = 0.0

        validation_loss = 0.0
        evaluation = []
        running_loss = 0.0

        num_correct = 0
        num_items = 0
        for _, data in enumerate(val_dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)[1]
            loss = loss_function(outputs, labels)
            validation_loss += loss.item()
            num_correct += (outputs.argmax(1) == labels).sum().cpu().item()
            num_items += len(outputs)

        validation_loss = validation_loss / len(val_dataset)
        accuracy = num_correct / num_items
        torch.save(
            model.state_dict(), os.path.join(save_dpath, f"resnet50_swav_{epoch}.pth")
        )
        print(
            "[Epoch %d / %d],  Validation loss: %.3f"
            % (epoch + 1, n_epochs, validation_loss)
        )
        writer.add_scalar("Running Loss/val", validation_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)
    writer.flush()
    writer.close()
    print("Finished Training")


if __name__ == "__main__":
    expname = datetime.datetime.now().replace(microsecond=0).isoformat()
    save_dpath = os.path.join(MODELS_DPATH, expname)
    os.makedirs(save_dpath)
    NUM_CLASSES = 7
    train_set, test_set = get_raf_dataloaders()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # TODO parametrable GPU number
    WEIGHT_PATH = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar"
    model = SwAV.load_from_checkpoint(WEIGHT_PATH, strict=True).model
    model.prototypes = nn.Linear(128, NUM_CLASSES)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train(
        dataset=train_set,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        n_epochs=N_EPOCHS,
        val_dataset=test_set,
        save_dpath=save_dpath,
        patience=PATIENCE,
        lr_decay=LR_DECAY
    )
