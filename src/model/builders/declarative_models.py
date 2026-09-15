import tensorflow as tf
from collections.abc import Callable
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from src.config.schemes.model_scheme import ModelConfig


def declarative_classification_model(
    cfg_model: ModelConfig,
    input_shape: tuple[int, int, int],
    num_classes: int,
    blocks_registry: dict[str, Callable[..., tf.Tensor]],
) -> Model:
    """Build a classification model by chaining configured model blocks.

    The architecture is defined declaratively through 'cfg_model.blocks'.
    Each block specifies a registered block type and its parameters. The
    corresponding block factory is resolved from 'blocks_registry' and
    applied sequentially to the model. The number of classes is injected
    automatically into the 'dense_head' block.

    Args:
        cfg_model: Model configuration containing the 'blocks' sequence.
            Each block must define a registered 'type' and optional
            'params'.
        input_shape: Input image shape as (height, width, channels).
        num_classes: Number of target classes for the output layer.
        blocks_registry: Registry mapping block type names to their factory
            functions.

    Returns:
        The constructed, uncompiled Keras model.
    """
    inputs = Input(shape=input_shape)

    x = inputs
    for block_cfg in cfg_model.blocks:

        block_fn = blocks_registry[block_cfg.type]
        params = dict(block_cfg.params)

        if block_cfg.type == "dense_head":
            params["num_classes"] = num_classes

        x = block_fn(x, **params)

    return Model(inputs=inputs, outputs=x)
