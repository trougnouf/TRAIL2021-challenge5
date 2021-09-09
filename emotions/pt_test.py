"""Test trained model on wikimedia_commons_emotions test set."""


import os
import random
import sys
import datetime
import yaml
import configargparse
import torch
from torch import nn
import torchvision
import numpy as np

sys.path.append("..")
from emotions import pt_common

random.seed(42)
CONFIG_FPATH = os.path.join("configs", "defaults_test.yaml")


def parse_arguments():
    """Parse config, return args."""
    parser = configargparse.ArgumentParser(
        description=__doc__,
        default_config_files=[CONFIG_FPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    pt_common.add_common_arguments(parser)
    return parser.parse_args()


# test(dataset=test_set, model=model, device=device, res=res)
def test(dataset, model, device, res, model_pretrain_method):
    """Test images, return results per class and overall."""
    print("Started testing")

    ncorrect_per_class = [0] * pt_common.NUM_CLASSES
    nsamples_per_class = [0] * pt_common.NUM_CLASSES
    for i, data in enumerate(dataset):
        # * ugly hack for bug https://github.com/PyTorchLightning/lightning-bolts/issues/731
        input, label = data
        input = input.to(device)
        label = label.to(device)
        input = torch.cat((input, input))  # *
        label = torch.cat((label, label))  # *
        output = pt_common.fw(model, input, model_pretrain_method)[0]
        output = output.argmax().item()  # *
        label = label[0].item()
        print(f"{label=}, {output=}")
        nsamples_per_class[label] += 1  # *
        if label == output:
            ncorrect_per_class[label] += 1
    for aclass in range(pt_common.NUM_CLASSES):
        res[aclass] = ncorrect_per_class[aclass] / nsamples_per_class[aclass]
    res["global"] = sum(ncorrect_per_class) / sum(nsamples_per_class)


if __name__ == "__main__":
    args = parse_arguments()
    assert args.pretrain_fpath is not None
    save_fpath = args.pretrain_fpath + f".test_{args.test_ds_names}.yaml"

    _, test_set = pt_common.get_dataloaders([], args.test_ds_names, batch_size=1)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = pt_common.init_model(args.model_pretrain_method, args.pretrain_url, device)
    pretrained_weights = torch.load(args.pretrain_fpath, map_location=device)
    model_weights = model.state_dict()
    model_weights.update(pretrained_weights)
    model.load_state_dict(model_weights, strict=True)
    model.eval()
    if os.path.isfile(save_fpath):
        with open(save_fpath, "r") as fp:
            res = yaml.safe_load(fp)
    else:
        res = dict()
    test(
        dataset=test_set,
        model=model,
        device=device,
        res=res,
        model_pretrain_method=args.model_pretrain_method,
    )
    print(res)
    with open(save_fpath, "w") as f:
        yaml.dump(res, f)
