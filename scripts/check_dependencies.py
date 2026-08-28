import hydra
import sklearn
import rasterio
import omegaconf
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

print("GPUS: ", tf.config.list_physical_devices("GPU"))
print("GPUS: ", tf.config.list_physical_devices("GPU").__len__() > 0)

print("Hydra version: ", hydra.__version__)
print("Scikit-learn version: ", sklearn.__version__)
print("Rasterio version: ", rasterio.__version__)
print("OmegaConf version: ", omegaconf.__version__)
print("matplotlib version: ", matplotlib.__version__)
print("Numpy version: ", np.__version__)
print("Pandas version: ", pd.__version__)
print("Tensorflow version: ", tf.__version__)
