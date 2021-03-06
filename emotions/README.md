# emotions

ResNet-50 models were trained to classify seven basic human emotions from datasets of faces.
The RAF basic dataset which consists of 15339 labeled face images was used to train all models. Some models were trained with KDEF as well while others left it available for testing.

# Data

raf_basic and kdef datasets are placed in ../datasets/train/[class]/[imagefiles] (and ../datasets/test/[class]/[imagefiles] for relevant raf_basic samples). A script will automatically handle this if the data is found in ../datasets/KDEF_and_AKDEF and ~/.cache/torch/mmf/data/raf_basic/basic

Classes are as follow:
1. Surprise
2. Fear
3. Disgust
4. Happiness
5. Sadness
6. Anger
7. Neutral

The cropped test data in ../datasets/test/wikimedia_commons_emotions is free (typically CC-BY-SA) but individual images' author information must be mentioned in publications. (This can be found by looking up the filenames, typically on https://commons.wikimedia.org )

# pip requirements:
pip install lightning-bolts ConfigArgParse PyYAML torchvision

# Training

`python3 pt_train.py --config configs/train_[simclr_ffhq, simclr_imagenet, simsiam_ffhq, swav_imagenet, torchvision_imagenet, weigthed_classes].yaml`

train_weighted_classes can be combined with other configuration, and yaml configuration files can be combined with command line arguments (which take priority).

You need to run `bash tools/download_pretrained_models.sh` before running a SimCLR or SimSiam based model.

# Testing

`python3 pt_test.py --config configs/test[simclr, simsiam, swav, torchvision].yaml --pretrain_fpath [path to trained model]`

# Results:

- train on raf and kdef, pretrain with SimCLR-FFHQ
  - raf test accuracy: .8243 (2h58 out of 2h58)
  - ../models/SimCLR_FFHQ/resnet50_swav_94.pth
  - wikimedia_commons_emotions: {0: 0.9166666666666666, 1: 0.16666666666666666, 2: 0.3333333333333333, 3: 0.4166666666666667, 4: 0.5, 5: 0.8333333333333334, 6: 0.16666666666666666, 'global': 0.47619047619047616}


- train on raf w/ balanced loss, pretrain with swav-ImageNet
  - raf test accuracy: .855 (1h52 out of 3h36)
  - ../models/emotions/Swav-ImageNet-rafonly-balanced/resnet50_swav_103.pth
  - wikimedia_commons_emotions: {0: 0.9166666666666666, 1: 0.08333333333333333, 2: 0.25, 3: 0.4166666666666667, 4: 0.6666666666666666, 5: 0.5, 6: 0.4166666666666667, 'global': 0.4642857142857143}
  - kdef: {0: 0.46131805157593125, 1: 0.008571428571428572, 2: 0.002857142857142857, 3: 0.07285714285714286, 4: 0.33476394849785407, 5: 0.025714285714285714, 6: 0.7314285714285714, 'global': 0.23381662242189094}


- train with raf and kdef, pretrain with swav-ImageNet
  - raf test accuracy: 0.8435 after 80 minutes (out of 180 minutes)
  - ../models/emotions/Swav-ImageNet/resnet50_swav_42.pth
    - wikimedia_commons_emotions: {0: 0.6666666666666666, 1: 0.4166666666666667, 2: 0.25, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.08333333333333333, 'global': 0.4166666666666667}

- train on raf and kdef, pretrain with SimSiam-FFHQ
  - raf test accuracy: .8044 (2h52 out of 2h57)
  - ../models/SimSiam_FFHQ/resnet50_swav_91.pth
  - wikimedia_commons_emotions: {0: 0.75, 1: 0.5833333333333334, 2: 0.16666666666666666, 3: 0.5, 4: 0.16666666666666666, 5: 0.5833333333333334, 6: 0.08333333333333333, 'global': 0.40476190476190477}


- train with raf, pretrain with swav-ImageNet
  - raf test accuracy: .8563 after 126 minutes (out of 140 minutes)
  - ../models/emotions/Swav-ImageNet-rafonly/resnet50_swav_116.pth
    - wikimedia_commons_emotions: {0: 0.9166666666666666, 1: 0.08333333333333333, 2: 0.08333333333333333, 3: 0.5833333333333334, 4: 0.4166666666666667, 5: 0.3333333333333333, 6: 0.4166666666666667, 'global': 0.40476190476190477}
    - kdef: {0: 0.4154727793696275, 1: 0.0, 2: 0.04428571428571428, 3: 0.08571428571428572, 4: 0.5350500715307582, 5: 0.007142857142857143, 6: 0.5214285714285715, 'global': 0.2297324892791505}


- train with raf and kdef from scratch
  - raf test accuracy: .7197 @ 89 (out of 2h40)
  - ../models/emotions/fromscratch/resnet50_swav_47.pth
    - wikimedia_commons_emotions: {0: 0.25, 1: 0.3333333333333333, 2: 0.08333333333333333, 3: 0.08333333333333333, 4: 0.3333333333333333, 5: 0.3333333333333333, 6: 0.08333333333333333, 'global': 0.21428571428571427}


- train on raf from scratch
  - raf test accuracy: .3862 after 0 epochs out of 3h36 (model reaches 100% train accuracy but test goes down)
  - ../models/emotions/fromscratch-rafonly/resnet50_swav_0.pth
  - wikimedia_commons_emotions: {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 'global': 0.14285714285714285}
  - kdef: {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 'global': 0.14294465999591588}


- Obsolete methods:
  - train w/ balanced dataset (removed samples):
    - raf test accuracy: .4736 after 8 minutes (out of 47 min)
    - ../models/emotions/2021-09-06T21:38:27/resnet50_swav_34.pth
      - wikimedia_commons_emotions: {0: 0.3333333333333333, 1: 0.9166666666666666, 2: 0.3333333333333333, 3: 0.0, 4: 0.25, 5: 0.5, 6: 0.0, 'global': 0.3333333333333333}

**TODO**:
- train w/ simclr weakly pretrained on imagenet (raf+kdef, raf)
- train w/ siamsiam weakly pretrained on imagenet (raf+kdef, raf)
