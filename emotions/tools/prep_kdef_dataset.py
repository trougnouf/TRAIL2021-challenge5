"""Copy RAF dataset from std path to ImageFolder-compatible structure."""

import os
import shutil
import pathlib
import sys

sys.path.append("..")
from emotions import pt_common

ORIG_DATA_DPATH = os.path.join(pt_common.DS_ROOT, "KDEF_and_AKDEF", "KDEF")
DS_NAME = "kdef"
LABELS_LUT = {"AF": 2, "AN": 6, "DI": 3, "HA": 4, "NE": 7, "SA": 5, "SU": 1}

#assert os.path.isdir(ORIG_DATA_DPATH), ORIG_DATA_DPATH


def make_KDEF_ImageFolder_struct():
    """
    Copy KDEF dataset from ORIG_DATADIR to ImageFolder-compatible structure.

    Results in DS_ROOT/[train or test]/kdef/[class_$n]/[fn]
    """
    dest_dpath = os.path.join(pt_common.DS_ROOT, "train", DS_NAME)
    for label in range(1, 8):
        os.makedirs(os.path.join(dest_dpath, str(label)), exist_ok=True)
    for aset in os.listdir(ORIG_DATA_DPATH):
        aset_dpath = os.path.join(ORIG_DATA_DPATH, aset)
        for fn in os.listdir(aset_dpath):
            src = os.path.join(aset_dpath, fn)
            try:
                dest = os.path.join(dest_dpath, str(LABELS_LUT[fn[4:6]]), fn)
            except KeyError as e:
                print(
                    f"make_KDEF_ImageFolder_struct warning: invalid fn: {src} with key {fn[4:6]}"
                )
                continue
            shutil.copyfile(src, dest)
            print(f"cp {src} {dest}")


if __name__ == "__main__":
    make_KDEF_ImageFolder_struct()
