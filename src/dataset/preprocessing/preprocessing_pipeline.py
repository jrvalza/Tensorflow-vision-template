from functools import partial
from omegaconf import DictConfig, OmegaConf

import tensorflow as tf

from .base_step_pipeline import BaseStepPipeline, Step


class PreprocessingPipeline(BaseStepPipeline):
    """Applies the steps configured in cfg.dataset.preprocessing.steps to a dataset."""

    def __init__(self, cfg_dataset: DictConfig) -> None:
        self._steps_registry: dict[str, Step] = {
            "pixel_value_normalization": self._pixel_value_normalization
        }
        super().__init__(cfg_dataset)

    def _config_entries(self, cfg_dataset: DictConfig) -> list[DictConfig]:
        return cfg_dataset.preprocessing.steps

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
        self, image: tf.Tensor, label: tf.Tensor, value: float = 255.0
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Scale image pixel values to [0, 1]."""
        return tf.cast(image / value, tf.float32), label
