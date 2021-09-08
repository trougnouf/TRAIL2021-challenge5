import torch.nn as nn
import torchvision
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised import SimCLR
from models.resnet50 import ResNet50Lightning


def load(nclasses, pretext, pretraining, batch_size):
    # Self supervised
    if pretraining == "swav":
        if pretext == "imagenet":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
            model = SwAV.load_from_checkpoint(weight_path, strict=True).model
            model.prototypes = nn.Linear(128, nclasses)
        else:
            raise NotImplementedError("Only 'imagenet' pretext task is currently supported")
    elif pretraining == "simclr":
        if pretext == "imagenet":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            model = simclr.encoder
            # model = nn.Sequential(
            #     simclr.encoder,
            #     Flatten(),
            #     nn.Linear(2048, nclasses)
            # )
        elif pretext == "chexpert":
            train_size = 65440
            model = SimCLR(num_samples=train_size, batch_size=batch_size, dataset="", gpus=1)
        else:
            raise NotImplementedError("Only 'imagenet' and 'chexpert' are already currently supported")

    # Suvervised
    elif pretext == "imagenet" and pretraining == "supervised":
        model = ResNet50Lightning(nclasses, pretrained=True)
        # model = torchvision.models.resnet50(pretrained=True)  # , num_classes=nclasses)
        #model.fc = nn.Linear(in_features=model.fc.in_features, out_features=nclasses)
    elif pretraining == "none" or pretraining is None:
        model = ResNet50Lightning(nclasses, pretrained=False)
        #model.fc = nn.Linear(in_features=model.fc.in_features, out_features=nclasses)
    else:
        raise NotImplementedError()
    return model
