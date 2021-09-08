import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import ImageOps, ImageEnhance, Image

import random
import math

"""
Transforamtions that we can apply on an image and the range of magnitude:

Rotate
Flip
Mirror
Equalize
Solarize, [0, 255]
Contrast, [0.1, 1.9]
Color, [0.1, 1.9]
Brightness, [0.1, 1.9]
Sharpness, [0.1, 1.9]
"""

def rotate(img, alpha):
    '''
    {'method': 'rotate', 'angle': alpha}
    '''
    return TF.rotate(img, alpha)

def flip(img, v):
    '''
    {'method': 'flip', 'value': v}
    '''
    return ImageOps.flip(img)

def mirror(img, v):
    '''
    {'method': 'mirror', 'value': v}
    '''
    return ImageOps.mirror(img)

def equalize(img, v):
    '''
    {'method': 'equalize'}
    '''

    return ImageOps.equalize(img)

def solarize(img, v):
    '''
    {'method': 'solarize', 'value': v}
    '''

    return ImageOps.solarize(img, v)

def contrast(img, v):
    '''
    {'method': 'contrast', 'value': v}
    '''

    return ImageEnhance.Contrast(img).enhance(v)

def color(img, v):
    '''
    {'method': 'color', 'value': v}
    '''

    return ImageEnhance.Color(img).enhance(v)

def brightness(img, v):
    '''
    {'method': 'brightness', 'value': v}
    '''

    return ImageEnhance.Brightness(img).enhance(v)

def sharpness(img, v):
    '''
    {'method': 'sharpness', 'value': v}
    '''

    return ImageEnhance.Sharpness(img).enhance(v)

def name_to_fct(method):

    op = {'rotate': rotate, 'flip': flip, 'mirror': mirror, 'equalize': equalize, 'solarize': solarize,
           'contrast': contrast, 'color': color, 'brightness': brightness, 'sharpness': sharpness}

    return op[method]


class RandAugmentation:

    def __init__(self):

        self.list = [(rotate, -30, 30), (mirror, 0, 1), (equalize, 0, 1), (solarize, 0, 255),
                     (contrast, 0.5, 1.9), (color, 0.1, 1.9), (brightness, 0.5, 1.9), (sharpness, 0.1, 1.9)]

        self.magnitude = 5

    def __call__(self, img):

        ops = random.choice(self.list)

        op, minv, maxv = ops
        
        if op.__name__ in ['flip', 'mirror', 'equalize']:
            img_ = op(img, 0)
            augment = {"method": op.__name__, "value": 0}

        else:
            val = random.uniform(minv, maxv)
            augment = {"method": op.__name__, "value": val}
            img_ = op(img, val)

        return img_, augment

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, ra=False, prob=0.5):
        self.augment = RandAugmentation()
        self.ra = ra
        self.prob = prob

    def __call__(self, sample):
        image = sample
        if self.ra:
            if random.random() < self.prob:
                augment_img = self.augment(image)
                return augment_img
        return sample

class Normalizer(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = TF.pil_to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        return image


class UnNormalizer(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            self.mean = mean
            self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor