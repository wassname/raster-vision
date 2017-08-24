#!/bin/bash

# This takes a GeoTIFF file and makes predictions in GeoJSON format.
# It's meant to be run on EC2 and downloads/uploads relevant files accordingly.

# Parse args
while :; do
    case $1 in
        --config-path)
            CONFIG_PATH=$2
            ;;
        --train-id)
            TRAIN_ID=$2
            ;;
        --checkpoint-id)
            CHECKPOINT_ID=$2
            ;;
        --tiff-id)
            TIFF_ID=$2
            ;;
        --dataset-id)
            DATASET_ID=$2
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)
            break
    esac
    shift
    shift
done

echo "CONFIG_PATH     = ${CONFIG_PATH}"
echo "TRAIN_ID        = ${TRAIN_ID}"
echo "CHECKPOINT_ID   = ${CHECKPOINT_ID}"
echo "TIFF_ID         = ${TIFF_ID}"
echo "DATASET_ID      = ${DATASET_ID}"

set -x
cd /opt/src/detection

S3_DATASETS=s3://raster-vision/datasets/detection
LOCAL_DATASETS=/opt/data/datasets/detection

S3_TRAIN=s3://raster-vision/results/detection/train
LOCAL_TRAIN=/opt/data/results/detection/train

S3_PREDICT=s3://raster-vision/datasets/detection/predict
LOCAL_PREDICT=/opt/data/datasets/detection/predict

# download tiff to run prediction on
TIFF_PATH=${LOCAL_PREDICT}/${TIFF_ID}/image.tif
aws s3 cp ${S3_PREDICT}/${TIFF_ID}/image.tif ${TIFF_PATH}

# download results of training
aws s3 sync ${S3_TRAIN}/${TRAIN_ID} ${LOCAL_TRAIN}/${TRAIN_ID}

# download training data and unzip (we just need the label map though...)
aws s3 cp ${S3_DATASETS}/${DATASET_ID}.zip ${LOCAL_DATASETS}
unzip -o ${LOCAL_DATASETS}/${DATASET_ID}.zip -d ${LOCAL_DATASETS}

# convert checkpoint to frozen inference graph
INFERENCE_GRAPH_PATH=${LOCAL_TRAIN}/${TRAIN_ID}/inference_graph.pb
python models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${CONFIG_PATH} \
    --checkpoint_path ${LOCAL_TRAIN}/${TRAIN_ID}/train/model.ckpt-${CHECKPOINT_ID} \
    --inference_graph_path ${INFERENCE_GRAPH_PATH}

# run sliding window over tiff to generate lots of window files
python scripts/make_windows.py \
    --image-path ${TIFF_PATH} \
    --output-dir ${TEMP}/windows \
    --window-size 300

# run prediction on the windows
LABEL_MAP_PATH=${LOCAL_DATASETS}/${DATASET_ID}/label_map.pbtxt
python scripts/predict.py \
    --frozen-graph-path ${INFERENCE_GRAPH_PATH} \
    --label-map-path ${LABEL_MAP_PATH} \
    --input-dir ${TEMP}/windows \
    --output-dir ${TEMP}/windows/predictions

# aggregate the predictions into an output geojson file
python scripts/aggregate_predictions.py \
    --image-path ${TIFF_PATH} \
    --window-info-path ${TEMP}/windows/window_info.json \
    --predictions-path ${TEMP}/windows/predictions/predictions.json \
    --label-map-path ${LABEL_MAP_PATH} \
    --output-dir ${LOCAL_PREDICT}/${TIFF_ID}/output

# upload the results to s3
aws s3 sync ${LOCAL_TRAIN}/${TRAIN_ID} ${S3_TRAIN}/${TRAIN_ID}
aws s3 sync ${LOCAL_PREDICT}/${TIFF_ID} ${S3_PREDICT}/${TIFF_ID}
