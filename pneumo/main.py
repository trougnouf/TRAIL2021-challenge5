#!/usr/bin/env python3
import argparse

from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised import SimCLR
from torch import nn
import torch
import datasets
import training
import models
import torchvision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretext", choices=["chestxray", "pneumonia",
                        "chexpert", "imagenet", "checkpoint", "none"], default=None)
    parser.add_argument("--pretraining", choices=["swav", "simclr", "supervised", "none"], default=None)
    parser.add_argument("--downstream", choices=["pneumonia", "chestxray", "none"], default=None)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    experiment_summary(args)
    return args


def experiment_summary(args):
    args.experiment_name = f"ds-{args.downstream}"
    if not (args.pretraining is None or args.pretraining == "none"):
        args.experiment_name = f"pretext_{args.pretext}_{args.pretraining}-{args.experiment_name}"

    print(f"""
    ##########################################################
    #                   Experiment summary                   #
    # Experiment name: {args.experiment_name}
    #--------------------------------------------------------#
    # Pretraining: {args.pretraining}
    # Pretext task: {args.pretext}
    # Checkpoint: {args.checkpoint}
    #--------------------------------------------------------#
    # Downstream task: {args.downstream}
    #--------------------------------------------------------#
    # Number of epochs: {args.nepochs}
    # Batch size: {args.batch_size}
    # Device: {args.device}
    # Log directory: {args.log_dir}
    ##########################################################
    """)


def pretext_task(model, task, pretraining, batch_size, device, nepochs, checkpoint, log_dir):
    if pretraining is None or pretraining == "none":
        return

    experiment_name = f"pretext_{task}_{pretraining}"
    if pretraining == "supervised":
        if task == "imagenet":
            model = torchvision.models.resnet50(pretrained=True)
        elif task == "checkpoint":
            model.load_state_dict(torch.load(checkpoint))
        else:
            train, test, _val = datasets.load(task, batch_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            device = torch.device(device)
            training.train_supervised(
                model,
                device,
                optimizer,
                nn.CrossEntropyLoss(),
                nepochs,
                train,
                test,
                experiment_name=experiment_name
            )

    elif pretraining == "simclr":
        if task == "imagenet":
            model = SimCLR.load_from_checkpoint(
                "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt")
            model = model.encoder
        elif task == "checkpoint":
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict, strict=False)
        else:
            train, test, _val = datasets.load(task, batch_size)
            training.train_ssl(model, train, test, device, nepochs, log_dir, experiment_name)

    elif pretraining == "swav":
        if task == "imagenet":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
            model = SwAV.load_from_checkpoint(weight_path, strict=True).model
        else:
            # WARNING: Untested !
            train, test, _val = datasets.load(task, batch_size)
            training.train_ssl(model, train, test, device, nepochs, log_dir, experiment_name)

    else:
        raise NotImplementedError()


def downstream_task(model, task, batch_size, device, nepochs, log_dir, experiment_name):
    if task is None or task == "none":
        return
    train, test, _val = datasets.load(task, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training.train_supervised(model, device, optimizer, nn.CrossEntropyLoss(), nepochs, train, test, experiment_name)


if __name__ == "__main__":
    args = parse_args()

    nclasses = 2 if args.downstream == "pneumonia" else 15
    model = models.load(
        nclasses,
        args.pretraining,
        args.batch_size
    )
    pretext_task(model, args.pretext, args.pretraining, args.batch_size,
                 args.device, args.nepochs, args.checkpoint, args.log_dir)
    if args.pretraining in ["simclr", "swav"]:
        model = models.add_classifier(model, args.downstream)
    experiment_name = f"pretext_{args.pretext}-pretraining_{args.pretraining}-ds_{args.downstream}"
    downstream_task(model, args.downstream, args.batch_size, args.device, args.nepochs, args.log_dir, experiment_name)
