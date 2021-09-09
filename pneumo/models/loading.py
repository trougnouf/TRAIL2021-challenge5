import torch.nn as nn
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised import SimCLR
from models.resnet50 import ResNet50Lightning
from models import pl_wrapper
import torchvision


def load(nclasses, pretraining, batch_size):
    if pretraining is None or pretraining == "none" or pretraining == "supervised":
        model = torchvision.models.resnet50(pretrained=False)
        num_filters = model.fc.in_features
        layers = list(model.children())[:-1]
        return nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(num_filters, nclasses),
            nn.Softmax(dim=1)
        )
        return ResNet50Lightning(nclasses, pretrained=False)
    elif pretraining == "simclr":
        return SimCLR(num_samples=0, batch_size=batch_size, dataset="", gpus=1)
    elif pretraining == "swav":
        return SwAV(num_samples=0, batch_size=batch_size, dataset="", gpus=1)
    else:
        raise NotImplementedError()


def add_classifier(ssl_model, downstream):
    if downstream == "pneumonia":
        num_classes = 2
    elif downstream == "chestxray":
        num_classes = 15
    # Replace the last layer with the amount of classes
    layers = list(ssl_model.encoder.children())[:-1]

    classifier = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(in_features=ssl_model.encoder.fc.in_features, out_features=num_classes),
        nn.Softmax()
    )
    return pl_wrapper.Wrapper(classifier)
