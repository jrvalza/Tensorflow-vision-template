
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Return the classification report as a dictionary"""
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
        )

def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    """Return the classification report as formatted text"""
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=False,
        zero_division=0
        )

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix"""
    return confusion_matrix(y_true, y_pred)

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the overall accuracy"""
    return round(accuracy_score(y_true, y_pred),4)

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the balanced accuracy"""
    return round(balanced_accuracy_score(y_true, y_pred), 4)

def compute_cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's kappa coefficient"""
    return round(cohen_kappa_score(y_true, y_pred),4)
