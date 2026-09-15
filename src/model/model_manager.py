from tensorflow.keras.models import Model

from src.config.registry import ModelRegistries
from src.config.schemes.model_scheme import ModelConfig


class ModelManager:
    """Builds Keras models using the configured model builder registry."""

    def __init__(
        self, cfg_model: ModelConfig, model_registries: ModelRegistries
    ) -> None:
        """Initialize the model manager.

        Args:
            cfg_model: Model configuration containing the selected builder
                and its parameters.
            model_registries: Registries containing the available model
                builders and architecture blocks.
        """
        self._cfg_model = cfg_model
        self._model_registries = model_registries

    def build(self, input_shape: tuple[int, int, int], num_classes: int) -> Model:
        """Build the configured Keras model.

        The model builder is selected from the registry using the configured
        builder name and receives the model configuration, input shape,
        number of classes, and registered architecture blocks.

        Args:
            input_shape: Input image shape as (height, width, channels).
            num_classes: Number of target classes passed to the model builder.

        Returns:
            The constructed, uncompiled Keras model.
        """
        build_fn = self._model_registries.builders[self._cfg_model.builder_name]

        return build_fn(
            self._cfg_model, input_shape, num_classes, self._model_registries.blocks
        )
