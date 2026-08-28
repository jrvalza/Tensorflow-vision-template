from collections.abc import Callable
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Dropout,
    Flatten,
    Resizing,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Layer,
)

POOLING_REGISTRY: dict[str, Callable[..., Layer]] = {
    "max2d": MaxPooling2D,
    "avg2d": AveragePooling2D,
}


def conv2d(
    x: tf.Tensor,
    filters: int,
    activation: str = "relu",
    batch_norm: bool = False,
    pooling: str | None = None,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """Apply a 2D convolution with optional normalization, pooling, and dropout.

    Args:
        x: Input tensor.
        filters: Number of convolutional filters.
        activation: Activation function used by the convolution.
        batch_norm: Whether to apply batch normalization.
        pooling: Optional pooling layer.
        dropout_rate: Dropout rate. A value of 0 disables dropout.

    Returns:
        Output tensor after applying the convolutional block.

    Raises:
        ValueError: If an unknown pooling layer is specified.
    """
    x = Conv2D(filters, (3, 3), padding="same", activation=activation)(x)

    if batch_norm:
        x = BatchNormalization()(x)
    if pooling:
        try:
            pooling_fn = POOLING_REGISTRY[pooling]
        except KeyError as e:
            raise ValueError(f"Unknown Pooling layer: {pooling}") from e

        x = pooling_fn(pool_size=(2, 2))(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x


def dense_head(
    x: tf.Tensor,
    num_classes: int,
    units: list[int],
    output_activation: str,
    batch_norm: bool = False,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """Build a fully connected classification head.

    Args:
        x: Input tensor.
        num_classes: Number of output classes.
        units: Number of units for each intermediate dense layer.
        output_activation: Activation function of the final classifier.
        batch_norm: Whether to apply batch normalization after each dense layer.
        dropout_rate: Dropout rate applied after each dense layer.

    Returns:
        Output tensor containing the class predictions.
    """
    x = Flatten()(x)
    for num_units in units:
        x = Dense(num_units, activation="relu")(x)

        if batch_norm:
            x = BatchNormalization()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    return Dense(num_classes, activation=output_activation)(x)


def vgg16_backbone(
    x: tf.Tensor,
    trainable_blocks: list[str],
    target_size: tuple[int, int],
    weights: str | None = "imagenet",
    include_top: bool = False,
) -> tf.Tensor:
    """Apply a VGG16 backbone with input adaptation and optional fine-tuning.

    Args:
        x: Input tensor.
        trainable_blocks: VGG16 layer-name prefixes to make trainable.
        target_size: Target spatial dimensions as (height, width).
        weights: Pre-trained weights to load, or random initialization.
        include_top: Whether to include the original VGG16 classification head.

    Returns:
        Output tensor produced by the VGG16 backbone.
    """
    x = Conv2D(
        filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu"
    )(x)
    x = Resizing(*target_size)(x)

    base_model = VGG16(
        input_shape=(*target_size, 3), weights=weights, include_top=include_top
    )

    for layer in base_model.layers:
        layer.trainable = any(
            layer.name.startswith(trainable_block)
            for trainable_block in trainable_blocks
        )
    return base_model(x)


BLOCKS_REGISTRY: dict[str, Callable[..., tf.Tensor]] = {
    "conv2d": conv2d,
    "dense_head": dense_head,
    "vgg16_backbone": vgg16_backbone,
}
