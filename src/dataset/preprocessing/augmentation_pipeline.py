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


class AugmentationPipeline:
    """Applies the transformations configured in cfg.dataset.augmentation.transforms to a dataset."""

    LAYER_REGISTRY: dict[str, Callable[..., Layer]] = {
        "random-flip": RandomFlip,
        "random-rotation": RandomRotation,
        "random-zoom": RandomZoom,
        "random-translation": RandomTranslation,
    }

    def __init__(self, cfg_dataset: DictConfig) -> None:
        self._layers: list[Layer] = [
            self._build_layer(transform_dict)
            for transform_dict in cfg_dataset.augmentation.transforms
        ]

    def _build_layer(self, transform_dict: DictConfig) -> Layer:
        """Instantiate the Keras layer configured in a single transform entry.

        Raises:
            ValueError: If transform name is not registered.
        """
        config = dict(OmegaConf.to_container(transform_dict, resolve=True))
        transform_name = config["name"]
        params = config.get("params", {})

        try:
            layer_cls = self.LAYER_REGISTRY[transform_name]
        except KeyError as e:
            raise ValueError(f"Unknown augmentation transform: {transform_name}") from e
        return layer_cls(**params)

    def apply(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply the configured augmentation layers to a dataset, in order"""
        if not self._layers:
            return dataset

        def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            for layer in self._layers:
                image = layer(image, training=True)
            return image, label

        return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
