from omegaconf import DictConfig

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History

from src.training.callbacks import resolve_callbacks
from src.training.optimizers import resolve_optimizer
from src.training.losses import resolve_loss


class Trainer:
    """Trainer class for training a Keras model"""

    def __init__(
        self,
        cfg: DictConfig,
        model: Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._history: History | None = None

    @property
    def history(self) -> History | None:
        """Training history from the last call to train()."""
        return self._history

    def train(self) -> Model:
        """Compile and train the model, returning the trained instance"""
        optimizer = resolve_optimizer(self._cfg.training)
        callbacks = resolve_callbacks(self._cfg.training)
        loss = resolve_loss(self._cfg.training)

        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=list(self._cfg.training.metrics),
        )

        self._history = self._model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=self._cfg.training.epochs,
            callbacks=callbacks,
        )
        return self._model
