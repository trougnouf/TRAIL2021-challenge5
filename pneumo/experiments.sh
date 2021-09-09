#! /bin/bash
epochs="20"
device="cuda"
batch_size="64"


for i in {0..3}; do
    ./main.py  \
        --pretext=imagenet \
        --pretraining=supervised \
        --downstream=chestxray \
        --nepochs=$epochs \
        --device=$device \
        --batch_size=$batch_size\
        --nepochs=$epochs \
        | tee runs/logs.txt
done;

