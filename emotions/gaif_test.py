import os
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import timm
import helpful_libs

import numpy as np

from PIL import Image
import sys

#CP_PATH = "weights/save_pneumonia_imagnet_resnet50"
CP_PATH = 'save/models/model_30000.ckpt'
CP_PATH = 'save/models/model_12000.ckpt'
ckpt = torch.load(CP_PATH)

new_ckpt = OrderedDict()

for key in ckpt['model'].keys():
    new_key = key[6:]
    new_ckpt[new_key] = ckpt['model'][key]

torch.save(new_ckpt, CP_PATH + "_test.pth")

model = timm.create_model(model_name="resnet50", checkpoint_path = CP_PATH + "_test.pth")

#fpath = sys.argv[1]
# infer
emotions_dict = {1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happiness', 5: 'Sadness', 6: 'Angry', 7: 'Neutral'}

test_dirs = ['test_images_cropped', '/home/imagedpt/.cache/torch/mmf/data/raf_basic/basic/Image/aligned']#, 'test_images']
#test_dirs = []
for test_dir in test_dirs:
    print(f'test_dir: {test_dir}')

    for fn in sorted(os.listdir(test_dir)):

        fpath = os.path.join(test_dir, fn)
        img_tensor = helpful_libs.fpath_to_tensor(fpath).unsqueeze(0)
        res = model(img_tensor)
        print(f'{fn}:\t {emotions_dict[res.argmax().item()+1]}; \t confidence = {res.max()}')
