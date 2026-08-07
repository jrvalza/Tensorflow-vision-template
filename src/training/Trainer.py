
from omegaconf import DictConfig

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History

from src.training.callbacks import resolve_callbacks
from src.training.optimizers import resolve_optimizer


class Trainer:
    """ Trainer class for training a Keras model"""

    def __init__(self, cfg: DictConfig, model: Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> None:
        self.cfg = cfg
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.history: History | None = None

    def train(self) -> Model:
        """ Compile and train a Keras model"""

        optimizer = resolve_optimizer(self.cfg.trainer)
        callbacks = resolve_callbacks(self.cfg.trainer)

        self.model.compile(
            optimizer=optimizer,
            loss=self.cfg.trainer.loss,
            metrics=list(self.cfg.trainer.metrics)
            )

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.trainer.epochs,
            callbacks=callbacks
        )
