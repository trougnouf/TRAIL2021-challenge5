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
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    args = parser.parse_args()
    experiment_summary(args)
    return args


def experiment_summary(args):
    print(f"""
    ##########################################################
    #                   Experiment summary                   #
    #--------------------------------------------------------#
    # Pretraining: {args.pretraining}
    # Pretext task: {args.pretext}
    #--------------------------------------------------------#
    # Downstream task: {args.downstream}
    # Number of epochs: {args.nepochs}
    #--------------------------------------------------------#
    # Log directory: {args.log_dir}
    # Checkpoint directory: {args.checkpoint_dir}
    # Metrics: {args.metrics}
    ##########################################################
    """)


if __name__ == "__main__":
    args = parse_args()

    nclasses = 2 if args.downstream == "pneumonia" else 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, loss_function = models.load(
        nclasses, device, args.pretext, args.pretraining)
    # Pretraining
    if args.pretraining != "none" and args.pretext != "imagenet":
        train, test, val = datasets.load(args.pretext)
        training.train(args.pretraining, model, device, optimizer,
                       loss_function, args.nepochs, train, test, args.metrics)

    # Downstream task
    print('Creating datasets')
    train, test = datasets.load(args.downstream)
    print('Dataset loaded')
    history = training.train("supervised", model, device, optimizer, loss_function,
                             args.nepochs, train, test, args.metrics)
