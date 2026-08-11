from omegaconf import DictConfig, OmegaConf

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from src.model.blocks import BLOCKS_REGISTRY


def declarative_model(
    cfg: DictConfig, input_shape: tuple[int, int, int], num_classes: int
) -> Model:
    """Create a model by chaining the blocks defined in cfg.model.blocks"""

    inputs = Input(shape=input_shape)

    x = inputs
    for block_cfg in cfg.model.blocks:
        try:
            block_fn = BLOCKS_REGISTRY[block_cfg.type]
        except KeyError as e:
            raise ValueError(f"Unknown block type: {block_cfg.type}") from e

        params = dict(OmegaConf.to_container(block_cfg.params, resolve=True))
        params = {
            k: (num_classes if v == "num_classes" else v) for k, v in params.items()
        }
        x = block_fn(x, **params)

    return Model(inputs=inputs, outputs=x)
