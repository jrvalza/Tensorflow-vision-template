from omegaconf import DictConfig

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History

from src.training.callbacks import resolve_callbacks
from src.training.optimizers import resolve_optimizer
from src.training.losses import resolve_loss


class Trainer:
    """Compiles and trains a Keras model using cfg.training (optimizer, loss, callbacks)."""

    def __init__(
        self,
        cfg_training: DictConfig,
        model: Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ) -> None:
        self._cfg_training = cfg_training
        self._model = model
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._history: History | None = None

    @property
    def history(self) -> History | None:
        """Training history from the last call to train(), or None if not trained yet."""
        return self._history

    def train(self) -> Model:
        """Compile and train the model according to cfg.training.

        Returns:
            The trained model instance.
        """
        optimizer = resolve_optimizer(self._cfg_training)
        callbacks = resolve_callbacks(self._cfg_training)
        loss = resolve_loss(self._cfg_training)

        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=list(self._cfg_training.metrics),
        )

        self._history = self._model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=self._cfg_training.epochs,
            verbose=self._cfg_training.verbose,
            callbacks=callbacks,
        )
        return self._model
