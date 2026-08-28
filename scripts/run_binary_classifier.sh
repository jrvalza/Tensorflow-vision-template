#!/bin/bash

python3 train.py \
  dataset.num_bands=3 \
  dataset.loader.params.batch_size=8 \
  dataset.loader.params.image_size=[64,64] \
  dataset.loader.params.label_mode="int" \
  training.loss.name="sparse_categorical_crossentropy" \
  "~training.loss.params.label_smoothing"