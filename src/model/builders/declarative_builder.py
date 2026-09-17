import tensorflow as tf
from collections.abc import Callable
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from src.config.schemes.model_scheme import ModelConfig


def declarative_builder(
    cfg_model: ModelConfig,
    input_shape: tuple[int, int, int],
    num_classes: int,
    blocks_registry: dict[str, Callable[..., tf.Tensor]],
) -> Model:
    """Build a model graph by chaining and connecting configured blocks.

    The architecture is defined declaratively through 'cfg_model.blocks'.
    Each block specifies a registered block type, its parameters, and a
    unique 'name'. By default a block is applied to the previous block's
    output; a block can instead pull an additional tensor from any
    earlier block by setting 'skip_from' to that block's name, which is
    passed to the block factory as the 'skip' argument (used e.g. by
    'upsample_concat_block' for U-Net-style skip connections).

    A block can request 'num_classes' to be injected into its params by
    setting 'inject_num_classes_as' to the target param name (e.g.
    'num_classes' for 'dense_head', or 'filters' for a 'conv2d' output
    layer in a segmentation head).

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

    named_outputs: dict[str, tf.Tensor] = {}

    for block_cfg in cfg_model.blocks:

        block_fn = blocks_registry[block_cfg.type]
        params = dict(block_cfg.params)

        if block_cfg.inject_num_classes_as:
            params[block_cfg.inject_num_classes_as] = num_classes

        if block_cfg.skip_connection_from:
            params["skip_connection"] = named_outputs[block_cfg.skip_connection_from]

        x = block_fn(x, **params)

        named_outputs[block_cfg.name] = x

    return Model(inputs=inputs, outputs=x)
