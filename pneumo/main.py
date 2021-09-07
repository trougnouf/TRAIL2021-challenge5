#!/usr/bin/env python3
import torch
import argparse
import datasets
import training
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretext", choices=["chestxray", "pneumonia", "chexpert", "imagenet", "none"], default=None)
    parser.add_argument("--pretraining", choices=["swav", "simclr", "supervised", "none"], default=None)
    parser.add_argument("--downstream", choices=["pneumonia", "chestxray"])
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--metrics", choices=["accuracy", "precision", "recall", "confusion_matrix"],
                        nargs="*", default=["accuracy", "precision", "recall", "confusion_matrix"])
    return parser.parse_args()


def experiment_summary(args):
    pass


if __name__ == "__main__":
    args = parse_args()

    use_pretrained_weights = (args.pretraining == "supervised" and args.pretext == "imagenet")
    nclasses = 2 if args.downstream == "pneumonia" else 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, loss_function = models.load(nclasses, use_pretrained_weights, device)

    # Pretraining
    if args.pretraining != "none" and args.pretext != "imagenet":
        train, test, val = datasets.load(args.pretext)
        training.train(args.pretraining, model, device, optimizer,
                       loss_function, args.nepochs, train, test, args.metrics)

    # Downstream task
    train, test, val = datasets.load(args.downstream)
    history = training.train("supervised", model, device, optimizer, loss_function,
                             args.nepochs, train, test, args.metrics)
