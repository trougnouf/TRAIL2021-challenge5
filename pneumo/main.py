#!/usr/bin/env python3
import torch
import argparse
import datasets
import training
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretext", choices=["chestxray", "pneumonia",
                        "chexpert", "imagenet", "none"], default=None)
    parser.add_argument("--pretraining", choices=["swav", "simclr", "supervised", "none"], default=None)
    parser.add_argument("--downstream", choices=["pneumonia", "chestxray", "none"], default=None)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
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
    # Batch size: {args.batch_size}
    ##########################################################
    """)


if __name__ == "__main__":
    args = parse_args()

    nclasses = 2 if args.downstream == "pneumonia" else 15
    model = models.load(
        nclasses,
        args.pretext,
        args.pretraining,
        args.batch_size
    )

    # Pretraining
    if args.pretraining is not None and args.pretraining != "none" and args.pretext != "imagenet":
        train, test, val = datasets.load(args.pretext, args.batch_size)
        training.train(model, train, test)

    # Downstream task
    train, test, val = datasets.load(args.downstream, args.batch_size)
    training.train(model, train, test)
