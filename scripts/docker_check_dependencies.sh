#!/bin/bash

docker compose run --rm tensorflow-ngc-2402 python3 -c "
import tensorflow as tf
import sklearn
import numpy as np
import matplotlib
import hydra
import omegaconf

print('Tensorflow version: ', tf.__version__)
print('GPUS: ', tf.config.list_physical_devices('GPU'))
print('Tiene GPU activa: ', len(tf.config.list_physical_devices('GPU')) > 0)
print('Scikit-learn version: ', sklearn.__version__)
print('Numpy version: ', np.__version__)
print('Matplotlib version: ', matplotlib.__version__)
print('Hydra version: ', hydra.__version__)
print('OmegaConf version: ', omegaconf.__version__)"