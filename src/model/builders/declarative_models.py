from omegaconf import DictConfig, OmegaConf

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from src.model.blocks import BLOCKS_REGISTRY


def declarative_classification_model(
    cfg: DictConfig, input_shape: tuple[int, int, int], num_classes: int
) -> Model:
    """Create a classification model by chaining the blocks defined in cfg.model.blocks"""

    inputs = Input(shape=input_shape)

    x = inputs
    for block_cfg in cfg.model.blocks:
        try:
            block_fn = BLOCKS_REGISTRY[block_cfg.type]
        except KeyError as e:
            raise ValueError(f"Unknown block type: {block_cfg.type}") from e

        params = dict(OmegaConf.to_container(block_cfg.params, resolve=True))

        if block_cfg.type == "dense_head":
            params["num_classes"] = num_classes

        x = block_fn(x, **params)

    return Model(inputs=inputs, outputs=x)
