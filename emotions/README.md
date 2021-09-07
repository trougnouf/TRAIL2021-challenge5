# Results:
- train with raf and kdef: 0.8435 after 80 minutes (out of 3h)
  - ../models/emotions/2021-09-06T18:36:17/resnet50_swav_42.pth
    - {0: 0.6666666666666666, 1: 0.4166666666666667, 2: 0.25, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.08333333333333333, 'global': 0.4166666666666667}
- train with raf: .8563 after 126 minutes (out of 140)
  - ../models/emotions/2021-09-06T15:50:10/resnet50_swav_116.pth
    - {0: 0.9166666666666666, 1: 0.08333333333333333, 2: 0.08333333333333333, 3: 0.5833333333333334, 4: 0.4166666666666667, 5: 0.3333333333333333, 6: 0.4166666666666667, 'global': 0.40476190476190477}
- train with raf and kdef from scratch: .7197 @ 89 (out of 2h40)
  - ../models/emotions/2021-09-07T10:40:44/resnet50_swav_47.pth
    - {0: 0.25, 1: 0.3333333333333333, 2: 0.08333333333333333, 3: 0.08333333333333333, 4: 0.3333333333333333, 5: 0.3333333333333333, 6: 0.08333333333333333, 'global': 0.21428571428571427}
    - {0: 0.3333333333333333, 1: 0.9166666666666666, 2: 0.3333333333333333, 3: 0.0, 4: 0.25, 5: 0.5, 6: 0.0, 'global': 0.3333333333333333}

- train w/ balanced dataset (removed samples): .4736 after 8 minutes (out of 47 min)
  - ../models/emotions/2021-09-06T21:38:27/resnet50_swav_34.pth
    - {0: 0.3333333333333333, 1: 0.9166666666666666, 2: 0.3333333333333333, 3: 0.0, 4: 0.25, 5: 0.5, 6: 0.0, 'global': 0.3333333333333333}


- train w/o pretrain on raf: .8103 after 50 minutes (out of 1h08); older method
sftp://multitel/home/imagedpt/doodling/save/logs/tensorboard_2021-09-02T17:02:55

# Data
Classes are as follow:
1: Surprise
2: Fear
3: Disgust
4: Happiness
5: Sadness
6: Anger
7: Neutral

The cropped test data in ../datasets/test/wikimedia_commons_emotions is free (typically CC-BY-SA) but author information must be mentioned in publications. (This can be found by looking up the filenames.)

# pip requirements:
pip install lightning-bolts ConfigArgParse PyYAML
