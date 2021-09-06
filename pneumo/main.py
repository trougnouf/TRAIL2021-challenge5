#!/usr/bin/env python3
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, CenterCrop
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from icecream import ic

from utils import Augmenter, Normalizer


def load_pneumonia(image_size=(224, 224), batch_size=32):
    train_set = ImageFolder(
        root="~/.cache/torch/mmf/data/pneumonia/train",
        transform=Compose([
            Resize(256),
            CenterCrop(224),
            Augmenter(ra=False, prob=0.5),
            ToTensor(),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    test_set = ImageFolder(
        root="~/.cache/torch/mmf/data/pneumonia/test",
        transform=Compose([
            Resize(256),
            CenterCrop(224),
            Augmenter(ra=False, prob=0.5),
            ToTensor(),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    val_set = ImageFolder(
        root="~/.cache/torch/mmf/data/pneumonia/val",
        transform=Compose([
            Resize(256),
            CenterCrop(224),
            Augmenter(ra=False, prob=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train, test, val


def validate(val_dataset, model, device, loss_function):
    validation_loss = 0.0
    num_correct = 0
    num_items = 0
    for _, data in enumerate(val_dataset):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        validation_loss += loss.item()
        num_correct += (outputs.argmax(1) == labels).sum().cpu().item()
        num_items += len(outputs)

    validation_loss = validation_loss / len(val_dataset)
    accuracy = num_correct / num_items
    return validation_loss, accuracy

def train(dataset, model, loss_function, optimizer, device, n_epochs, log_dir=None):
    print("Started training")
    writer = SummaryWriter('runs/sup_pretrained_'+str(np.random.randint(0, 100)))
    history = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        evaluation = []
        for i, data in enumerate(dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            evaluation.append((outputs.argmax(1) == labels).cpu().sum().item() / len(outputs))
            running_loss += loss.item()
            if i % 10 == 9:
                running_loss = running_loss / 10
                accuracy = np.mean(evaluation)
                log_step = epoch * len(dataset) + i
                history.append((log_step, running_loss, accuracy))
                print('[%d, %5d/%d] loss: %.3f\t accuracy: %.3f' % (epoch + 1, i + 1, len(dataset), running_loss, accuracy))
                writer.add_scalar("Running Loss/train", running_loss, log_step)
                writer.add_scalar("Accuracy/train", accuracy, log_step)
                evaluation = []
                running_loss = 0.0
        val_loss, val_accuracy = validate(val_set, model, device, loss_function)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
    writer.flush()
    writer.close()
    print('Finished Training')
    return history

def train_avegraged(n, dataset, n_epochs, log_dir):
    histories = []
    for _ in range(n):
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())   
        histories.append(train(dataset, model, loss_function, optimizer, device, n_epochs, log_dir=None))
    steps = np.array([history[0] for history in histories[0]])
    losses = np.array([[h[1] for h in history] for history in histories])
    accuracies = np.array([[h[2] for h in history] for history in histories])
    mean_accuray = accuracies.mean(axis=0)
    mean_loss = losses.mean(axis=0)

    writer = SummaryWriter(log_dir)
    for s, a, l in zip(steps, mean_accuray, mean_loss):
        writer.add_scalar("Accuracy/train", a, s)
        writer.add_scalar("Running Loss/train", l, s)
    writer.flush()
    writer.close()
    print("Done")



if __name__ == "__main__":
    train_set, test_set, val_set = load_pneumonia()
    train_avegraged(10, train_set, n_epochs=20, log_dir="runs/baseline_pneumonia_resnet50")
