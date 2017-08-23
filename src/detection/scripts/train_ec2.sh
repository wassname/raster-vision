#!/bin/bash

DATA=$1
# DATA=singapore_ships_chips_tiny
CONFIG=$2
# CONFIG=/opt/src/detection/config/ssd_mobilenet_v1_pets.config
RUN=$3
# RUN="pets0"
SYNC_INTERVAL="15m"

cd /opt/src/detection

# sync results of previous run just in case it crashed in the middle of running
rm -R /opt/data/results/detection/$RUN
aws s3 sync s3://raster-vision/results/detection/$RUN /opt/data/results/detection/$RUN

# download pre-trained model (to use as starting point) and unzip
aws s3 cp s3://raster-vision/datasets/detection/models/ssd_mobilenet_v1_coco_11_06_2017.zip /opt/data/datasets/detection/models/
unzip -o /opt/data/datasets/detection/models/ssd_mobilenet_v1_coco_11_06_2017.zip -d /opt/data/datasets/detection/models/

# download training data and unzip
aws s3 cp s3://raster-vision/datasets/detection/${DATA}.zip /opt/data/datasets/detection/
unzip -o /opt/data/datasets/detection/${DATA}.zip -d /opt/data/datasets/detection/

/opt/src/detection/scripts/s3_sync.sh $SYNC_INTERVAL $RUN &

# run 3 tf scripts including tensorboard on port 6006
mkdir -p /opt/data/results/detection/$RUN
mkdir -p /opt/data/results/detection/$RUN/train
mkdir -p /opt/data/results/detection/$RUN/eval

python models/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=$CONFIG \
    --train_dir=/opt/data/results/detection/$RUN/train \
& \
python models/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=$CONFIG \
    --checkpoint_dir=/opt/data/results/detection/$RUN/train \
    --eval_dir=/opt/data/results/detection/$RUN/eval \
& \
tensorboard --logdir=/opt/data/results/detection/$RUN/
