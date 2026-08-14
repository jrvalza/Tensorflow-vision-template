from typing import TypeAlias
from omegaconf import DictConfig
from collections.abc import Callable

import tensorflow as tf

PreprocessingStep: TypeAlias = Callable[
    [tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]
]


class PreprocessingPipeline:
    """Applies the steps configured in cfg.dataset.preprocessing.steps to a dataset."""

    def __init__(self, cfg_dataset: DictConfig) -> None:
        self._cfg_dataset = cfg_dataset
        self._steps_registry: dict[str, PreprocessingStep] = {
            "pixel-value-normalization": self._pixel_value_normalization
        }

    def _resolve_step(self, step_name: str) -> PreprocessingStep:
        """Resolve a preprocessing step name to its corresponding function.

        Raises:
            ValueError: If step_name is not registered.
        """
        try:
            return self._steps_registry[step_name]
        except KeyError as e:
            raise ValueError(f"Unknown preprocessing step: {step_name}") from e

    def _pixel_value_normalization(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Normalize image pixel values by scaling them from [0, 255] to [0, 1]."""
        image = tf.cast(image / 255.0, tf.float32)
        return image, label

    def apply(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply the configured preprocessing pipeline to a dataset."""
        process_fns: list[PreprocessingStep] = [
            self._resolve_step(step_name)
            for step_name in self._cfg_dataset.preprocessing.steps
        ]

        if not process_fns:
            return dataset

        def preprocess(
            image: tf.Tensor, label: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            for fn in process_fns:
                image, label = fn(image, label)
            return image, label

        return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
