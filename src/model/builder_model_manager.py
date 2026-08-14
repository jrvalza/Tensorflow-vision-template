import json
from typing import Callable
from omegaconf import DictConfig
from tensorflow.keras.models import Model
from .builders.deep_cnn import deep_cnn
from .builders.declarative_models import declarative_classification_model


class BuilderModelManager:
    """Builds a Keras model from the architecture configured in cfg.model.builder."""

    BUILDER_REGISTRY: dict[str, Callable[..., Model]] = {
        "deep_cnn": deep_cnn,
        "declarative_classification_model": declarative_classification_model,
    }

    def __init__(self, cfg_model: DictConfig) -> None:
        self._cfg_model = cfg_model

    def __str__(self) -> str:
        """List the available model builders."""
        models = {
            model_type: model_cls.__name__
            for model_type, model_cls in self.BUILDER_REGISTRY.items()
        }
        return f"Available model builders:\n{json.dumps(models, indent=4)}"

    def build(self, input_shape: tuple[int, int, int], num_classes: int) -> Model:
        """Build the model configured in cfg.model.builder.

        Args:
            input_shape: Input image shape as (height, width, channels).
            num_classes: Number of target classes, forwarded to the builder.

        Returns:
            The constructed, uncompiled Keras model.

        Raises:
            ValueError: If cfg.model.builder is not registered.
        """
        try:
            build_fn = self.BUILDER_REGISTRY[self._cfg_model.builder]
        except KeyError as e:
            raise ValueError(f"Unknown model builder: {self._cfg_model.builder}") from e
        return build_fn(self._cfg_model, input_shape, num_classes)
