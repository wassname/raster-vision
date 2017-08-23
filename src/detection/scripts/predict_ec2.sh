#!/bin/bash

CONFIG=$1
# CONFIG=/opt/src/detection/configs/ssd_mobilenet_v1_pets.config
RUN=$2
# RUN="pets0"
CHECKPOINT_NUMBER=$3
# CHECKPOINT_NUMBER=135656
TIFF=$4
DATA=$5

cd /opt/src/detection

# download tiff to run prediction on
# note that tiff is in results/detection/predictions, not datasets.
aws s3 cp s3://raster-vision/results/detection/predictions/$TIFF/image.tif \
    /opt/data/results/detection/predictions/$TIFF

# download results of training
aws s3 sync s3://raster-vision/results/detection/$RUN \
    /opt/data/results/detection/$RUN

# download training data and unzip (we just need the label map though...)
aws s3 cp s3://raster-vision/datasets/detection/${DATA}.zip /opt/data/datasets/detection/
unzip -o /opt/data/datasets/detection/${DATA}.zip -d /opt/data/datasets/detection/

# convert checkpoint to frozen inference graph
python models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $CONFIG \
    --checkpoint_path /opt/data/results/detection/$RUN/train/model.ckpt-${CHECKPOINT_NUMBER} \
    --inference_graph_path /opt/data/results/detection/$RUN/inference_graph.pb

# run sliding window over tiff to generate lots of window files
python scripts/make_windows.py \
    --image-path /opt/data/results/detection/predictions/$TIFF/image.tif \
    --output-dir $TMP/windows \
    --window-size 300

# run prediction on the windows
python scripts/predict.py \
    --frozen_graph_path=/opt/data/results/detection/$RUN/inference_graph.pb \
    --label_map_path=/opt/data/datasets/detection/$DATA/label_map.pbtxt \
    --input_dir=$TMP/windows
    --output_dir=$TMP/windows/predictions

# aggregate the predictions into an output geojson file
python scripts/aggregate_predictions.py \
    --image-path /opt/data/results/detection/predictions/$TIFF/image.tif \
    --window-info-path $TMP/windows/window_info.json \
    --predictions-path $TMP/windows/predictions/predictions.json \
    --label-map-path /opt/data/datasets/detection/$DATA/label_map.pbtxt \
    --output-dir /opt/data/results/detection/predictions/$TIFF/output

# upload the results to s3
aws s3 sync /opt/data/results/detection/$RUN s3://raster-vision/results/detection/$RUN
aws s3 sync /opt/data/results/detection/predictions/$TIFF/output s3://raster-vision/results/detection/predictions/$TIFF/output
