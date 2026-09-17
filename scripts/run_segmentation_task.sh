#!/bin/bash

python3 train.py \
  task="segmentation" \
  dataset.dataset_name="aerial_images" \
  dataset.root_dir="raw_data/segmentation" \
  dataset.num_bands=3 \
  dataset.image_dtype="uint8" \
  dataset.loader.name="from_csv_metadata" \
  dataset.loader.params.batch_size=16 \
  dataset.loader.params.label_mode="int" \
  dataset.loader.params.image_size=[128,128] \
  dataset.loader.params.validation_split=0.2 \
  model=unet_segmentation \
  training.metrics=["accuracy"] \
  training.loss.name="sparse_categorical_crossentropy" \
  "~training.loss.params.label_smoothing" \
  training.optimizer.name="adam" \
  training.optimizer.params.learning_rate=0.0001 \
  "$@"