import json
from abc import ABC, abstractmethod

import tensorflow as tf
from omegaconf import DictConfig
from tensorflow.keras.models import Model

from src.utils.paths import get_evaluation_dir


class BaseEvaluator(ABC):
    """Base interface for task-specific model evaluation"""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def _save_report(self, report: dict, filename: str) -> None:
        """Persist the classification report as JSON"""
        report_path = get_evaluation_dir() / filename
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    @abstractmethod
    def evaluate(
        self, model: Model, test_ds: tf.data.Dataset, class_names: list[str]
    ) -> None:
        """Evaluate the model on the test dataset according to cfg.model.task."""
        pass
