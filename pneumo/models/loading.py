import torch
import torch.nn as nn
import torchvision
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised import SimCLR
from models.layers import Flatten


def load(nclasses, device, pretext, pretraining):
    if pretext == "imagenet" and pretraining == "swav":
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
        model = SwAV.load_from_checkpoint(weight_path, strict=True).model
        model.prototypes = nn.Linear(128, nclasses)
    # elif pretraining == "swav":
    #     model = SwAV(
    #         gpus=1,
    #         num_samples=len(train_set),
    #         dataset='x',
    #         batch_size=32,
    #         queue_path='runs',
    #     )
    elif pretext == "imagenet" and pretraining == "simclr":
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        model = nn.Sequential(
            simclr.encoder,
            Flatten(),
            nn.Linear(2048, nclasses))
    elif pretext == "imagenet" and pretraining == "supervised":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=nclasses)
    elif pretext == "none" or pretext is None:
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=nclasses)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer, nn.CrossEntropyLoss()
