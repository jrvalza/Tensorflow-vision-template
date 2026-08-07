
import json
from omegaconf import DictConfig
from tensorflow.keras.models import Model
from .builders.deep_cnn import deep_cnn


class ModelManager:
    """Manager class for model creation from the project configuration."""

    MODEL_REGISTRY = {
        "deep_cnn": deep_cnn
    }

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def __str__(self) -> str:
        """Return the available model architectures."""

        models = {model_type: model_cls.__name__ for model_type, model_cls in self.MODEL_REGISTRY.items()}
        return f"Available model architecture:\n{json.dumps(models, indent=4)}"

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
            build_fn = self.MODEL_REGISTRY[self._cfg.model.architecture]
        except KeyError as e:
            raise ValueError(f"Unknown model architecture: {self._cfg.model.architecture}") from e
        return build_fn(self._cfg, input_shape, num_classes) 
    