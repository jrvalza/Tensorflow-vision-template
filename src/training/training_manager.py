import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import History, Callback

from src.utils.savers import save_figure
from src.config.registry import TrainingRegistries
from src.config.schemes.training_scheme import TrainingConfig


class TrainingManager:
    """Compiles and trains a Keras model using the configured training components."""

    def __init__(
        self,
        cfg_training: TrainingConfig,
        model: Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        training_registries: TrainingRegistries,
    ) -> None:
        """Initialize the training manager.

        Args:
            cfg_training: Training configuration containing the optimizer,
                loss, metrics, callbacks, epochs, and verbosity settings.
            model: Keras model to compile and train.
            train_ds: Training dataset used to fit the model.
            val_ds: Validation dataset used during training.
            training_registries: Registries containing the available
                optimizers, losses, and callbacks.
        """
        self._cfg_training = cfg_training
        self._model = model
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._training_registries = training_registries
        self._history: History | None = None

    def _plot_training_curves(self) -> None:
        """Plot and save training and validation curves for each metric.

        Curves are generated from the history produced by 'model.fit()'.
        Metrics with a corresponding validation metric are plotted together.
        The resulting figure is saved as 'training_curves.png'.

        If training history is not available, no plot is generated.
        """
        if self._history is None:
            return

        metrics = [
            metric for metric in self._history.history if not metric.startswith("val_")
        ]

        fig, axes = plt.subplots(
            1, len(metrics), figsize=(6 * len(metrics), 5), sharex=True
        )

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):

            ax.plot(self._history.history[metric], label="Training", linewidth=1)

            val_metric = f"val_{metric}"
            if val_metric in self._history.history:
                ax.plot(
                    self._history.history[val_metric], label="Validation", linewidth=1
                )

            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())

            if "accuracy" in metric:
                ax.set_ylim(0, 1.1)

            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        fig.suptitle("Training History", fontsize=14, fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "training_curves.png")

    def _resolve_optimizer(self) -> Optimizer:
        """Instantiate the configured Keras optimizer.

        The optimizer class is resolved from the registry and initialized
        with the parameters defined in the training configuration.

        Returns:
            An instantiated Keras optimizer.
        """
        optimizer_cls = self._training_registries.optimizers[
            self._cfg_training.optimizer.name
        ]
        params = dict(self._cfg_training.optimizer.params)
        return optimizer_cls(**params)

    def _resolve_callbacks(self) -> list[Callback]:
        """Instantiate the configured Keras callbacks.

        Callbacks are resolved from the registry and instantiated with their
        configured parameters, preserving their order in the configuration.

        Returns:
            A list of instantiated Keras callbacks in configuration order.
        """
        callbacks = []
        for cfg_callback in self._cfg_training.callbacks:

            callback_cls = self._training_registries.callbacks[cfg_callback.name]
            params = dict(cfg_callback.params)
            callbacks.append(callback_cls(**params))

        return callbacks

    def _resolve_loss(self) -> Loss:
        """Instantiate the configured Keras loss.

        The loss class is resolved from the registry and initialized with
        the parameters defined in the training configuration.

        Returns:
            An instantiated Keras loss.
        """
        loss_cls = self._training_registries.losses[self._cfg_training.loss.name]
        params = dict(self._cfg_training.loss.params)
        return loss_cls(**params)

    def train(self) -> Model:
        """Compile and train the configured Keras model.

        The optimizer, loss, and callbacks are resolved from their registries,
        after which the model is compiled and trained using the configured
        training and validation datasets. The training history is stored
        internally and used to generate the training curves.

        Returns:
            The trained Keras model.
        """
        optimizer = self._resolve_optimizer()
        callbacks = self._resolve_callbacks()
        loss = self._resolve_loss()

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

        self._plot_training_curves()

        return self._model
