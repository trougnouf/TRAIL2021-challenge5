"""Train loop for RAF model with PyTorch Lightning."""
​
import os
import random
import sys
import datetime
import yaml
import configargparse
import torch
from torch import nn
import numpy as np
from pl_bolts.models.self_supervised import SwAV
​
sys.path.append("..")
from emotions import pt_common
​
CONFIG_FPATH = os.path.join("configs", "defaults.yaml")
random.seed(42)
​
​
def parse_arguments():
    """Parse config, return args."""
    parser = configargparse.ArgumentParser(
        description=__doc__,
        default_config_files=[CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    pt_common.add_common_arguments(parser)
    parser.add_argument("--n_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--patience", type=int, help="How many epochs before LR decays")
    parser.add_argument(
        "--lr_decay",
        type=float,
        help="LR decay (multiply LR by this every [patience] epochs.)",
    )
    parser.add_argument(
        "--train_ds_names",
        nargs="+",
        help="Dataset names (directories in ../datasets/emotions/train/)",
    )
    parser.add_argument(
        "--lossf",
        help="Loss function",
    )
    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="Train from scratch",
    )
    parser.add_argument(
        "--expname", help="experiment name (default is autogenerated from datetime)"
    )
    parser.add_argument(
        "--loss_weights",
        nargs="*",
        type=float,
        help="(space-separated) Classes weights used in the training loss. Set to 0 for uniform weights.",
    )
    return parser.parse_args()
​
​
def weights_init(m):
    """Weights initializer used when --no_pretrain is set."""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
​
​
def train(
    dataset,
    model,
    loss_function,
    optimizer,
    device,
    n_epochs,
    val_dataset,
    save_dpath,
    patience,
    lr_decay,
):
    """Training loop."""
    print("Started training")
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(save_dpath, "tensorboard")
    )
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # simple lr decay every patience epochs
        if epoch % patience == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay
                print(f'train: reduced lr to {param_group["lr"]}')
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        running_loss = 0.0
        evaluation = []
        for i, data in enumerate(dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
​
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)[1]
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
​
            evaluation.append((outputs.argmax(1) == labels).cpu().sum().item())
​
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                running_loss = running_loss / 10
                print(
                    "[%d, %5d/%d] loss: %.3f"
                    % (epoch + 1, i + 1, len(dataset), running_loss)
                )
                writer.add_scalar(
                    "Running Loss/train", running_loss, epoch * len(dataset) + i
                )
                writer.add_scalar(
                    "Accuracy/train", np.mean(evaluation) / 32, epoch * len(dataset) + i
                )
                evaluation = []
                running_loss = 0.0
​
        validation_loss = 0.0
        evaluation = []
        running_loss = 0.0
​
        num_correct = 0
        num_items = 0
        for _, data in enumerate(val_dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
​
            # forward
            outputs = model(inputs)[1]
            loss = loss_function(outputs, labels)
            validation_loss += loss.item()
            num_correct += (outputs.argmax(1) == labels).sum().cpu().item()
            num_items += len(outputs)
​
        validation_loss = validation_loss / len(val_dataset)
        accuracy = num_correct / num_items
        torch.save(
            model.state_dict(), os.path.join(save_dpath, f"resnet50_swav_{epoch}.pth")
        )
        print(
            "[Epoch %d / %d],  Validation loss: %.3f"
            % (epoch + 1, n_epochs, validation_loss)
        )
        writer.add_scalar("Running Loss/val", validation_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)
    writer.flush()
    writer.close()
    print("Finished Training")
​
​
if __name__ == "__main__":
    args = parse_arguments()
    if not args.expname:
        args.expname = datetime.datetime.now().replace(microsecond=0).isoformat()
    save_dpath = os.path.join(pt_common.MODELS_DPATH, args.expname)
    os.makedirs(save_dpath, exist_ok=True)
    with open(os.path.join(save_dpath, "config.yaml"), "w") as fp:
        yaml.dump(vars(args), fp)
​
    train_set, test_set = pt_common.get_dataloaders(
        args.train_ds_names, args.test_ds_names
    )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = SwAV.load_from_checkpoint(args.pretrain_url, strict=True).model
    model.prototypes = nn.Linear(128, pt_common.NUM_CLASSES)
    model = model.to(device)
​
    if args.no_pretrain:
        print("Reseting weights.")
        model.apply(weights_init)
    if (args.lossf == "CrossEntropy" and len(args.loss_weights) == pt_commons.NUM_CLASSES):
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(args.loss_weights).to(device)
        )
    elif args.lossf == "CrossEntropy":
        print('pt_train: using CrossEntropyLoss with uniform weights.')
        loss_function = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(args.lossf)
    optimizer = torch.optim.Adam(model.parameters())
    train(
        dataset=train_set,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        n_epochs=args.n_epochs,
        val_dataset=test_set,
        save_dpath=save_dpath,
        patience=args.patience,
        lr_decay=args.lr_decay,
    )
