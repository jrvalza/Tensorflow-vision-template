import numpy as np
from typing import Any

from src.utils.savers import save_report
from src.evaluation.evaluators.base_evaluator import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    """Evaluate a classification model: report + confusion matrix."""

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """Compute the task-specific metrics, later used by '_report'/'_plot'."""
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "report": self._metrics.classification_report_dict(
                y_true, y_pred, self._class_names
            ),
            "overall_accuracy": self._metrics.compute_accuracy(y_true, y_pred),
            "balanced_accuracy": self._metrics.compute_balanced_accuracy(
                y_true, y_pred
            ),
            "cohen_kappa": self._metrics.compute_cohen_kappa(y_true, y_pred),
            "confusion_matrix": self._metrics.compute_confusion_matrix(y_true, y_pred),
        }

    def _report(self, results: dict[str, Any]) -> None:
        """Print and persist the computed results."""
        print(
            self._metrics.classification_report_text(
                results["y_true"], results["y_pred"], self._class_names
            )
        )
        print(f"Overall Accuracy : {results['overall_accuracy']}")
        print(f"Balanced Accuracy : {results['balanced_accuracy']}")
        print(f"Cohen's Kappa    : {results['cohen_kappa']}")

        save_report(
            {
                "overall_accuracy": results["overall_accuracy"],
                "balanced_accuracy": results["balanced_accuracy"],
                "cohen_kappa": results["cohen_kappa"],
                "classification_report": results["report"],
            },
            filename="classification_report.json",
        )

    def _plot(self, results: dict[str, Any]) -> None:
        """Generate and save the evaluation plots."""
        self._plotter.plot_confusion_matrix(
            results["confusion_matrix"], self._class_names
        )
