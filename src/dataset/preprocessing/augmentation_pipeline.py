from collections.abc import Callable
from omegaconf import DictConfig, OmegaConf

import tensorflow as tf
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    Layer,
)
from .base_step_pipeline import BaseStepPipeline, Step


class AugmentationPipeline(BaseStepPipeline):
    """Applies the transforms configured in cfg.dataset.augmentation.transforms to a dataset."""

    LAYER_REGISTRY: dict[str, Callable[..., Layer]] = {
        "random_flip": RandomFlip,
        "random_rotation": RandomRotation,
        "random_zoom": RandomZoom,
        "random_translation": RandomTranslation,
    }

    def __init__(self, cfg_dataset: DictConfig) -> None:
        super().__init__(cfg_dataset)

    def _config_entries(self) -> list[DictConfig]:
        return self._cfg_dataset.augmentation.transforms

    def _resolve_entry(self, entry: DictConfig) -> Step:
        """
        Raises:
            ValueError: If transform name is not registered.
        """
        config = dict(OmegaConf.to_container(entry, resolve=True))
        transform_name = config["name"]
        params = config.get("params", {})

        try:
            layer_cls = self.LAYER_REGISTRY[transform_name]
        except KeyError as e:
            raise ValueError(f"Unknown augmentation transform: {transform_name}") from e

        layer = layer_cls(**params)

        def step(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return layer(image, training=True), label

        return step
