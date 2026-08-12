import json
from typing import Callable
from omegaconf import DictConfig
from tensorflow.keras.models import Model
from .builders.deep_cnn import deep_cnn
from .builders.declarative_models import declarative_classification_model


class BuilderModelManager:
    """Manager class for model creation from the project configuration."""

    BUILDER_REGISTRY: dict[str, Callable] = {
        "deep_cnn": deep_cnn,
        "declarative_classification_model": declarative_classification_model,
    }

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def __str__(self) -> str:
        """Return the available model builders"""
        models = {
            model_type: model_cls.__name__
            for model_type, model_cls in self.BUILDER_REGISTRY.items()
        }
        return f"Available model builders:\n{json.dumps(models, indent=4)}"

    def build(self, input_shape: tuple[int, int, int], num_classes: int) -> Model:
        """
        Build the configured model architecture.

        Args:
            input_shape: Shape of the input images.
            num_classes: Number of output classes.

        Returns:
            A Keras model.
        """
        try:
            build_fn = self.BUILDER_REGISTRY[self._cfg.model.builder]
        except KeyError as e:
            raise ValueError(f"Unknown model builder: {self._cfg.model.builder}") from e
        return build_fn(self._cfg, input_shape, num_classes)
