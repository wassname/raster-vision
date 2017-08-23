### Overview

### Data Prep

Generate chips and convert to TFRecord format.

```
python scripts/tiff_chipper.py \
    --tiff-path /opt/data/datasets/detection/singapore_ships/0.tif \
    --json-path /opt/data/datasets/detection/singapore_ships/0.geojson \
    --output-dir /opt/data/datasets/detection/singapore_ships_chips_tiny \
    --chip-size 300 --debug
```

```
python scripts/create_tf_record.py \
    --data_dir=/opt/data/datasets/detection/singapore_ships_chips_tiny \
    --output_dir=/opt/data/datasets/detection/singapore_ships_chips_tiny \
    --label_map_path=/opt/data/datasets/detection/singapore_ships_chips_tiny/label_map.pbtxt
```

### Training a model on EC2
First, create a TF config file `configs/ships/ssd_mobilenet_v1.config`
which is used to configure the training.
Then, start a training job on AWS Batch, by running the following from the VM
```
src/detection/scripts/batch_submit.py lf/train-ships \
    /opt/src/detection/scripts/train_ec2.sh \
    singapore_ships_chips_tiny.zip configs/ships/ssd_mobilenet_v1.config ships0
```
You can view the progress of the training using Tensorboard by pointing your browser at `<ec2 instance ip>:6006`. When you are satisfied with the results, you need to kill the job since it's running in an infinite loop. Recent model checkpoints are synced to the S3 bucket under `results/detection/ships0`.

In order to use the model for prediction, you will first need to convert a checkpoint file to a frozen inference graph.

```

```

### Making predictions for individual images on EC2
To start a prediction job, you can run
```
src/detection/scripts/batch_submit.py lf/train-ships \
    /opt/src/detection/scripts/predict_ec2.sh \
    /opt/src/detection/configs/ssd_mobilenet_v1_pets.config pets0 135656
```
which will put predictions in the S3 bucket in `results/detection/pets0/predictions`.

### Making predictions for a large image with many objects locally
The neural network can only handle small images with a fixed size. Therefore, for large images which contain many objects, we use the following strategy. First, we slide a window over the large image and generate a directory
full of window images. Then, we run the usual predict script over that directory
of images. Finally, we aggregate the predictions on the windows, taking into
account the offset of each window within the larger image.

The following commands assume that you have the frozen inference graph at `/opt/data/results/detection/pets0/output_inference_graph.pb` and the [animal montage image](src/detection/img/animal_montage.jpg) at `/opt/data/datasets/detection/pets/images_subset/animal_montage.jpg`.
Running these should generate output at `/opt/data/results/detection/windows`.
```
python scripts/make_windows.py \
    --image-path /opt/data/datasets/detection/pets/images_subset/animal_montage.jpg \
    --output-dir /opt/data/results/detection/windows \
    --window-size 300

python scripts/predict.py \
    --frozen_graph_path=/opt/data/results/detection/pets0/output_inference_graph.pb \
    --label_map_path=/opt/data/datasets/detection/pets/pet_label_map.pbtxt \
    --input_dir=/opt/data/results/detection/windows/windows \
    --output_dir=/opt/data/results/detection/windows/predictions

python scripts/aggregate_predictions.py \
    --image-path /opt/data/datasets/detection/pets/images_subset/animal_montage.jpg \
    --window-info-path /opt/data/results/detection/windows/window_info.json \
    --predictions-path /opt/data/results/detection/windows/predictions/predictions.json \
    --label-map-path /opt/data/datasets/detection/pets/pet_label_map.pbtxt \
    --output-dir /opt/data/results/detection/windows
```
Due to the sliding window approach, sometimes there are multiple detections where there should be one, so we group them using a clustering algorithm in OpenCV. There is an `eps` parameter in `detection/scripts/aggregate_predictions.py` that will probably need to be tuned further depending on the dataset. Here are the predictions before and after grouping.
![Predictions on animal montage](img/animal_montage_predictions.jpg)
![Predictions on animal montage with detection grouping](img/animal_montage_predictions2.jpg)
