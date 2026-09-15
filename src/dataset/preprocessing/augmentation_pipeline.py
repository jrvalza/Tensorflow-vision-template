from collections.abc import Callable

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .base_pipeline import BasePipeline, Step
from src.config.schemes.dataset_scheme import DatasetConfig
from src.config.schemes.dataset_scheme import AugmentationStepConfig


class AugmentationPipeline(BasePipeline):
    """Applies the configured data augmentation transforms to a dataset.

    Each transform is instantiated from the provided registry and applied
    to the input image during pipeline execution. Labels are passed through
    unchanged.
    """

    def __init__(
        self,
        cfg_dataset: DatasetConfig,
        transforms_registry: dict[str, Callable[..., Layer]],
    ) -> None:
        """Initialize the augmentation pipeline.

        Args:
            cfg_dataset: Configuration containing the augmentation
                transforms and their parameters.
            transforms_registry: Registry mapping transform names to Keras layer
                classes.
        """
        self._transforms_registry = transforms_registry
        super().__init__(cfg_dataset)

    def _config_entries(self) -> list[AugmentationStepConfig]:
        return self._cfg_dataset.augmentation.transforms

    def _resolve_entry(self, entry: AugmentationStepConfig) -> Step:
        layer_cls = self._transforms_registry[entry.name](**entry.params)

        def step(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return layer_cls(image, training=True), label

        return step
