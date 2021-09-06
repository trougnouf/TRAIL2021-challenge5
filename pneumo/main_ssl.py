import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from pl_bolts.models.self_supervised import SwAV
from datetime import datetime


def load_pneumonia(image_size=(224, 224), batch_size=32):
    train_set = ImageFolder(
        root="~/.cache/torch/mmf/data/pneumonia/train",
        transform=Compose([
            Resize(image_size),
            ToTensor()
        ])
    )
    test_set = ImageFolder(
        root="~/.cache/torch/mmf/data/pneumonia/test",
        transform=Compose([
            Resize(image_size),
            ToTensor()
        ])
    )
    val_set = ImageFolder(
        root="~/.cache/torch/mmf/data/pneumonia/val",
        transform=Compose([
            Resize(image_size),
            ToTensor()
        ])
    )
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_set, batch_size=batch_size)
    return train, test, val


def train(dataset, model, loss_function, optimizer, device, n_epochs, val_dataset):
    print("Started training")
    writer = SummaryWriter('runs/resnet50_Swav_'+str(datetime.now()))
    for epoch in range(n_epochs):  # loop over the dataset multiple times
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
                print('[%d, %5d/%d] loss: %.3f' % (epoch + 1, i + 1, len(dataset), running_loss))
                writer.add_scalar("Running Loss/train", running_loss, epoch * len(dataset) + i)
                writer.add_scalar("Accuracy/train", np.mean(evaluation)/32, epoch * len(dataset) + i)
                evaluation = []
                running_loss = 0.0

        validation_loss = 0.0
        evaluation = []
        for i, data in enumerate(val_dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)[1]
            loss = loss_function(outputs, labels)

            evaluation.append((outputs.argmax(1) == labels).cpu().sum().item())

            # print statistics
            validation_loss += loss.item()
            if i % 5 == 4:
                validation_loss = validation_loss / 10
                print('[%d, %5d/%d] Validation loss: %.3f' % (epoch + 1, i + 1, len(dataset), validation_loss))
                writer.add_scalar("Running Loss/val", validation_loss, epoch * len(dataset) + i)
                writer.add_scalar("Accuracy/val", np.mean(evaluation)/32, epoch * len(dataset) + i)
                evaluation = []
                validation_loss = 0.0
    writer.flush()
    writer.close()
    print('Finished Training')


if __name__ == "__main__":
    num_classes = 2
    train_set, test_set, val_set = load_pneumonia()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    model = SwAV.load_from_checkpoint(weight_path, strict=True).model
    model.prototypes=nn.Linear(128, num_classes)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train(train_set, model, loss_function, optimizer, device, 20, test_set)
