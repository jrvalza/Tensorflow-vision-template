from functools import partial
from omegaconf import DictConfig, OmegaConf

import tensorflow as tf

from .base_step_pipeline import BaseStepPipeline, Step


class PreprocessingPipeline(BaseStepPipeline):
    """Applies the steps configured in cfg.dataset.preprocessing.steps to a dataset."""

    def __init__(self, cfg_dataset: DictConfig) -> None:
        self._steps_registry: dict[str, Step] = {
            "pixel_value_normalization": self._pixel_value_normalization,
            "resize_image": self._resize_image,
        }
        super().__init__(cfg_dataset)

    def _config_entries(self) -> list[DictConfig]:
        return self._cfg_dataset.preprocessing.steps

    def _resolve_entry(self, entry: DictConfig) -> Step:
        """
        Raises:
            ValueError: If step name is not registered.
        """
        config = dict(OmegaConf.to_container(entry, resolve=True))
        step_name = config["name"]
        params = config.get("params", {})

        try:
            step = self._steps_registry[step_name]
        except KeyError as e:
            raise ValueError(f"Unknown preprocessing step: {step_name}") from e
        return partial(step, **params)

    def _pixel_value_normalization(
        self, image: tf.Tensor, label: tf.Tensor, dtype: str
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Scale image pixel values to [0, 1]."""
        image_float = tf.cast(image, tf.float32)
        image_norm = image_float / tf.cast(tf.as_dtype(dtype).max, tf.float32)
        return image_norm, label

    def _resize_image(
        self, image: tf.Tensor, label: tf.Tensor, image_size: list[int]
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Resize image to the specified size."""
        resized_image = tf.image.resize(
            image, image_size, method=tf.image.ResizeMethod.BILINEAR
        )

        if self._cfg_dataset.task == "classification":
            return resized_image, label

        elif self._cfg_dataset.task == "segmentation":
            resized_mask = tf.image.resize(
                label, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )

            resized_mask = tf.cast(resized_mask, tf.int32)

            return resized_image, resized_mask
