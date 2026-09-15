import numpy as np
from typing import Any


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


class Metrics:
    """Provides metric computations for model evaluation."""

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute the confusion matrix.

        Args:
            y_true: Ground-truth class indices, shape (N,).
            y_pred: Predicted class indices, shape (N,).

        Returns:
            Confusion matrix of shape (num_classes, num_classes), rows as
            true labels and columns as predicted labels.
        """
        return confusion_matrix(y_true, y_pred)

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the overall classification accuracy.

        Args:
            y_true: Ground-truth class indices, shape (N,).
            y_pred: Predicted class indices, shape (N,).

        Returns:
            Overall classification accuracy.
        """
        return round(accuracy_score(y_true, y_pred), 4)

    def compute_balanced_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute the balanced classification accuracy.

        Args:
            y_true: Ground-truth class indices, shape (N,).
            y_pred: Predicted class indices, shape (N,).

        Returns:
            Balanced classification accuracy.
        """
        return round(balanced_accuracy_score(y_true, y_pred), 4)

    def compute_cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Cohen's kappa coefficient.

        Args:
            y_true: Ground-truth class indices, shape (N,).
            y_pred: Predicted class indices, shape (N,).

        Returns:
            Cohen's kappa coefficient.
        """
        return round(cohen_kappa_score(y_true, y_pred), 4)

    def classification_report_text(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
    ) -> str:
        """Return the classification report as formatted text.

        Args:
            y_true: Ground-truth class indices, shape (N,).
            y_pred: Predicted class indices, shape (N,).
            class_names: Names of the target classes.

        Returns:
            Classification report formatted as a string.
        """
        return classification_report(
            y_true, y_pred, target_names=class_names, output_dict=False, zero_division=0
        )

    def classification_report_dict(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
    ) -> dict[str, Any]:
        """Return the classification report as a dictionary.

        Args:
            y_true: Ground-truth class indices, shape (N,).
            y_pred: Predicted class indices, shape (N,).
            class_names: Names of the target classes.

        Returns:
            Classification report containing precision, recall, F1-score,
            and support for each class.
        """
        return classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
