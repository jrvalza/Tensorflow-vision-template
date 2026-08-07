
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model

from src.evaluation.metrics import (
    classification_report_dict,
    classification_report_text,
    compute_accuracy,
    compute_balanced_accuracy,
    compute_cohen_kappa,
    compute_confusion_matrix
)

from src.evaluation.plots import plot_confusion_matrix
from src.evaluation.evaluators.BaseEvaluator import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    """Evaluate a classification model: report + confusion matrix."""

    def _predict_labels(self, model: Model, test_ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Run inference over a batched dataset"""

        y_true, y_pred = [], []

        for images, labels in test_ds:

            y_pred.append(np.argmax(model.predict(images, verbose=0), axis=-1))

            if self.cfg.dataset.label_mode == "categorical":
                y_true.append(np.argmax(labels.numpy(), axis=-1))
            else:
                y_true.append(labels.numpy().squeeze())

        return np.concatenate(y_true), np.concatenate(y_pred)

    def evaluate(self, model: Model, test_ds: tf.data.Dataset, class_names: list[str]) -> None:
        """Evaluate the model on the test dataset"""

        y_true, y_pred = self._predict_labels(model, test_ds)
 
        report = classification_report_dict(y_true, y_pred, class_names)
        oa = compute_accuracy(y_true, y_pred) 
        b_acc = compute_balanced_accuracy(y_true, y_pred)
        kappa = compute_cohen_kappa(y_true, y_pred)
        print(classification_report_text(y_true, y_pred, class_names))
        print(f"Overall Accuracy : {oa}")
        print(f"Balanced Accuracy : {b_acc}")
        print(f"Cohen's Kappa    : {kappa}")

        self._save_report({"overall_accuracy": oa,
                           "balanced_accuracy": b_acc,
                           "cohen_kappa": kappa,
                           "classification_report": report
                           }
        )

        cm = compute_confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, class_names)
