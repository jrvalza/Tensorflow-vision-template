from omegaconf import DictConfig, OmegaConf

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from src.model.blocks import BLOCKS_REGISTRY


def declarative_classification_model(
    cfg_model: DictConfig, input_shape: tuple[int, int, int], num_classes: int
) -> Model:
    """Build a classification model by chaining the blocks in cfg.model.blocks.

    Args:
        cfg_model: 'model' section of the config. blocks must be a list of {type, params}
            entries, type being a key in BLOCKS_REGISTRY. num_classes is
            injected automatically into dense_head params.
        input_shape: Input image shape as (height, width, channels).
        num_classes: Number of target classes for the output layer.

    Returns:
        The constructed, uncompiled Keras model.

    Raises:
        ValueError: If a block type is not registered.
    """
    inputs = Input(shape=input_shape)

    x = inputs
    for block_cfg in cfg_model.blocks:
        try:
            block_fn = BLOCKS_REGISTRY[block_cfg.type]
        except KeyError as e:
            raise ValueError(f"Unknown block type: {block_cfg.type}") from e

        params = dict(OmegaConf.to_container(block_cfg.params, resolve=True))

        if block_cfg.type == "dense_head":
            params["num_classes"] = num_classes

        x = block_fn(x, **params)

    return Model(inputs=inputs, outputs=x)
