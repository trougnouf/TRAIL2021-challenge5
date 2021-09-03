import torch
import os
import cv2
import numpy as np

def fpath_to_tensor(img_fpath, device=torch.device(type='cpu'), batch=False):
    tensor = torch.tensor(img_path_to_np_flt(img_fpath), device=device)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor

def img_path_to_np_flt(fpath):
    '''returns a numpy float32 array from RGB image path (8-16 bits per component)
    shape: c, y, x
    FROM common.libimgops'''
    if not os.path.isfile(fpath):
        raise ValueError(f'File not found {fpath}')
    try:
        rgb_img = cv2.cvtColor(cv2.imread(fpath, flags=cv2.IMREAD_COLOR+cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB).transpose(2,0,1)
    except cv2.error as e:
        print(f'img_path_to_np_flp: error {e} with {fpath}')
        breakpoint()
    if rgb_img.dtype == np.ubyte:
        return rgb_img.astype(np.single)/255
    elif rgb_img.dtype == np.ushort:
        return rgb_img.astype(np.single)/65535
    else:
        raise TypeError("img_path_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})")

