from typing import TypeAlias
from collections.abc import Callable

from abc import ABC, abstractmethod
import tensorflow as tf
from omegaconf import DictConfig

Step: TypeAlias = Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]


class BaseStepPipeline(ABC):
    """Base class for pipelines that apply a configured sequence of (image, label) steps."""

    def __init__(self, cfg_dataset: DictConfig) -> None:
        self._cfg_dataset = cfg_dataset
        self._steps: list[Step] = [
            self._resolve_entry(entry) for entry in self._config_entries()
        ]

    @abstractmethod
    def _config_entries(self) -> list[DictConfig]:
        """Return this pipeline's list of {name, params} entries from cfg_dataset."""

    @abstractmethod
    def _resolve_entry(self, entry: DictConfig) -> Step:
        """Turn a single {name, params} config entry into a callable step."""

    def apply(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply the configured steps to a dataset, in order."""
        if not self._steps:
            return dataset

        def run(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            for step in self._steps:
                image, label = step(image, label)
            return image, label

        return dataset.map(run, num_parallel_calls=tf.data.AUTOTUNE)
