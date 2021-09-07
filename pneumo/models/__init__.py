from .utils import StochasticDepth, stochastic_depth
import torch
import torchvision
import torch.nn as nn


def load(nclasses, pretrained, device):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=nclasses)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer, nn.CrossEntropyLoss()
