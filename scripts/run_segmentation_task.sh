#!/bin/bash

python3 train.py \
  task="segmentation" \
  dataset.dataset_name="aerial_images" \
  dataset.root_dir="raw_data/segmentation" \
  dataset.num_bands=3 \
  dataset.loader.name="from_csv_metadata" \
  dataset.loader.params.batch_size=16 \
  dataset.loader.params.image_size=[64,64] \
  dataset.loader.params.label_mode="int" \
  training.loss.name="sparse_categorical_crossentropy" \
  "~training.loss.params.label_smoothing" \
  "$@"