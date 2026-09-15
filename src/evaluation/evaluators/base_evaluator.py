from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from ..metrics import Metrics
from ..plotter import Plotter


class BaseEvaluator(ABC):
    """Abstract base class for task-specific model evaluators.

    Implements the evaluation flow as a template method: 'evaluate()' is
    fixed and common to every task. Subclasses only customize behavior
    through the '_compute_metrics', '_report' and '_plot' hooks.
    """

    def __init__(
        self,
        model: Model,
        test_ds: tf.data.Dataset,
        class_names: list[str],
        label_mode: str,
        metrics: Metrics,
        plotter: Plotter,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Trained Keras model to evaluate.
            test_ds: Batched test dataset used for evaluation.
            class_names: Class names, in the order used by the model's output.
            label_mode: Label encoding used by the dataset, such as
                'categorical', 'binary', or 'int'.
            metrics: Metrics object used to compute evaluation metrics.
            plotter: Plotter object used to generate evaluation plots.
        """
        self._model = model
        self._test_ds = test_ds
        self._class_names = class_names
        self._label_mode = label_mode
        self._metrics = metrics
        self._plotter = plotter

    @abstractmethod
    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """Compute the task-specific metrics, later used by '_report'/'_plot'."""
        ...

    @abstractmethod
    def _report(self, results: dict[str, Any]) -> None:
        """Print and persist the computed results."""
        ...

    @abstractmethod
    def _plot(self, results: dict[str, Any]) -> None:
        """Generate and save the evaluation plots."""
        ...

    def _predict_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on the test dataset and return class indices.

        Model predictions are converted to class indices using 'argmax'.
        Ground-truth labels are converted from one-hot encoding when
        'label_mode' is 'categorical'; otherwise, their existing class
        indices are used.

        Returns:
            A tuple containing the ground-truth and predicted class indices
            as flattened NumPy arrays.
        """
        y_true, y_pred = [], []

        for images, labels in self._test_ds:

            y_pred.append(np.argmax(self._model.predict(images, verbose=0), axis=-1))

            if self._label_mode == "categorical":
                y_true.append(np.argmax(labels.numpy(), axis=-1))
            else:
                y_true.append(labels.numpy().squeeze())

        return np.concatenate(y_true), np.concatenate(y_pred)

    def evaluate(self) -> None:
        """Run the fixed evaluation flow: predict, compute, report, plot."""
        y_true, y_pred = self._predict_labels()
        results = self._compute_metrics(y_true, y_pred)
        self._report(results)
        self._plot(results)
