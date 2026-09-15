import tensorflow as tf
from typing import TypeAlias
from abc import ABC, abstractmethod
from collections.abc import Callable

from src.config.schemes.dataset_scheme import (
    PreprocessingStepConfig,
    AugmentationStepConfig,
)
from src.config.schemes.dataset_scheme import DatasetConfig

Step: TypeAlias = Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]


class BasePipeline(ABC):
    """Abstract base class for dataset transformation pipelines.

    Defines the common interface and execution logic for pipelines that
    apply a configured sequence of transformations to image-label pairs.
    Concrete pipelines are responsible for resolving their configuration
    entries into callable steps.
    """

    def __init__(self, cfg_dataset: DatasetConfig) -> None:
        """Initialize the pipeline by resolving its configured steps.

        Args:
            cfg_dataset: Dataset configuration from which '_config_entries()'
                reads the steps to resolve into callables.
        """
        self._cfg_dataset = cfg_dataset
        self._steps: list[Step] = [
            self._resolve_entry(entry) for entry in self._config_entries()
        ]

    @abstractmethod
    def _config_entries(
        self,
    ) -> list[PreprocessingStepConfig] | list[AugmentationStepConfig]:
        """Return the configured pipeline steps.

        Returns:
            A list of configuration entries defining the steps to apply.
            Each entry is expected to contain a step name and its parameters.
        """
        ...

    @abstractmethod
    def _resolve_entry(
        self, entry: PreprocessingStepConfig | AugmentationStepConfig
    ) -> Step:
        """Resolve a configuration entry into a callable step.

        Args:
            entry: Configuration entry containing the step name and its
                parameters.

        Returns:
            A callable transformation that accepts an image-label pair
            and returns the transformed image-label pair.
        """
        ...

    def apply(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply the configured steps to a dataset in sequence.

        Each step receives the output of the previous step, preserving
        the order defined in the pipeline configuration.

        Args:
            dataset: TensorFlow dataset containing image-label pairs.

        Returns:
            A dataset with the configured transformations applied.
        """
        if not self._steps:
            return dataset

        def run(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            for step in self._steps:
                image, label = step(image, label)
            return image, label

        return dataset.map(run, num_parallel_calls=tf.data.AUTOTUNE)
