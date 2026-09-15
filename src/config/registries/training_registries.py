"""Central registry of training optimizers, losses, and callbacks.

To add a new optimizer, loss, or callback, register it here under a unique
string key. That key is what users reference from YAML config, and is
validated against these registries by the corresponding configuration schemas.

"""

from tensorflow.keras.optimizers import SGD, Adam, AdamW, Optimizer
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from tensorflow.keras.losses import (
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    Loss,
)

TRAINING_OPTIMIZERS_REGISTRY: dict[str, type[Optimizer]] = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}

TRAINING_LOSSES_REGISTRY: dict[str, type[Loss]] = {
    "categorical_crossentropy": CategoricalCrossentropy,
    "sparse_categorical_crossentropy": SparseCategoricalCrossentropy,
}

TRAINING_CALLBACKS_REGISTRY: dict[str, type[Callback]] = {
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
}
