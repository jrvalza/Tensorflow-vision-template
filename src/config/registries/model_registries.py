"""Central registry of model builders and reusable architecture blocks.

To add a new model builder or block, implement it in 'src/model' and
register it here under a unique string key. That key is what users
reference from YAML config ('builder_name' / 'blocks[].type'), and is
validated against this registry by 'src.config.schemes.model_scheme'.
"""

from collections.abc import Callable

import tensorflow as tf
from tensorflow.keras.models import Model

from src.model.blocks import (
    conv2d,
    pooling2d,
    dense_head,
    vgg16_backbone,
    conv2d_transpose,
    upsample_concat_block,
    MODEL_POOLING_LAYERS_REGISTRY,
)
from src.model.builders.declarative_builder import declarative_builder

MODEL_BUILDERS_REGISTRY: dict[str, Callable[..., Model]] = {
    "declarative_builder": declarative_builder,
}


MODEL_BLOCKS_REGISTRY: dict[str, Callable[..., tf.Tensor]] = {
    "conv2d": conv2d,
    "pooling2d": pooling2d,
    "dense_head": dense_head,
    "vgg16_backbone": vgg16_backbone,
    "conv2d_transpose": conv2d_transpose,
    "upsample_concat_block": upsample_concat_block,
}
