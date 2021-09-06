"""Copy RAF dataset from std path to ImageFolder-compatible structure."""

import os
import shutil
import pathlib
import env

ORIG_DATADIR = os.path.join(
    pathlib.Path.home(),
    ".cache",
    "torch",
    "mmf",
    "data",
    "raf_basic",
    "basic",
    "Image",
    "aligned",
)
DS_NAME = "raf_basic"
LABELS_FPATH = os.path.join(
    pathlib.Path.home(),
    ".cache",
    "torch",
    "mmf",
    "data",
    "raf_basic",
    "basic",
    "EmoLabel",
    "list_patition_label.txt",
)


def make_raf_ImageFolder_struct():
    """
    Copy RAF dataset from ORIG_DATADIR to ImageFolder-compatible structure.

    Results in DS_ROOT/[train or test]/kdef/[class_n]/[fn]
    """
    for label in range(1, 8):
        os.makedirs(
            os.path.join(env.DS_ROOT, "train", DS_NAME, str(label)), exist_ok=True
        )
        os.makedirs(
            os.path.join(env.DS_ROOT, "test", DS_NAME, str(label)), exist_ok=True
        )
    with open(LABELS_FPATH, "r") as fp:
        for line in fp:
            fn, label = line[:-1].split(" ")
            fn_actual = fn[:-4] + "_aligned" + ".jpg"
            train_or_test = fn.split("_")[0]
            src = os.path.join(ORIG_DATADIR, fn_actual)
            dest = os.path.join(env.DS_ROOT, train_or_test, DS_NAME, label, fn_actual)
            shutil.copyfile(src, dest)
            print(f"cp {src} {dest}")


if __name__ == "__main__":
    make_raf_ImageFolder_struct()
