from collections.abc import Callable
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    concatenate,
    Conv2DTranspose,
    Dropout,
    Flatten,
    Resizing,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Layer,
)
from tensorflow.keras.applications import VGG16

MODEL_POOLING_LAYERS_REGISTRY: dict[str, Callable[..., Layer]] = {
    "max2d": MaxPooling2D,
    "avg2d": AveragePooling2D,
}


def pooling2d(
    x: tf.Tensor,
    method: str = "max2d",
    pool_size: tuple[int, int] = (2, 2),
) -> tf.Tensor:
    """Apply a standalone pooling layer.

    Args:
        x: Input tensor.
        method: Registered pooling layer name (see MODEL_POOLING_LAYERS_REGISTRY).
        pool_size: Pooling window size.

    Returns:
        Output tensor after pooling.

    Raises:
        ValueError: If an unknown pooling layer is specified.
    """
    try:
        pooling_fn = MODEL_POOLING_LAYERS_REGISTRY[method]
    except KeyError as e:
        raise ValueError(f"Unknown pooling layer: {method}") from e
    return pooling_fn(pool_size=pool_size)(x)


def conv2d(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (1, 1),
    padding: str = "same",
    activation: str = "relu",
    batch_norm: bool = False,
    pooling: str | None = None,
    pool_size: int | tuple[int, int] = (2, 2),
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """Apply a 2D convolution with optional normalization, pooling, and dropout.

    Args:
        x: Input tensor.
        filters: Number of convolutional filters.
        kernel_size: Size of the convolution kernel.
        strides: Stride of the convolution.
        padding: Padding mode used by the convolution.
        activation: Activation function used by the convolution.
        batch_norm: Whether to apply batch normalization.
        pooling: Optional pooling layer.
        pool_size: Size of the pooling window.
        dropout_rate: Dropout rate. A value of 0 disables dropout.

    Returns:
        Output tensor after applying the convolutional block.

    Raises:
        ValueError: If 'pooling' does not correspond to a registered
            pooling layer..
    """
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
    )(x)

    if batch_norm:
        x = BatchNormalization()(x)
    if pooling is not None:
        x = pooling2d(x, method=pooling, pool_size=pool_size)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x


def conv2d_transpose(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (2, 2),
    strides: tuple[int, int] = (2, 2),
    padding: str = "same",
    hidden_activation: str | None = None,
) -> tf.Tensor:
    """Apply a transposed (upsampling) 2D convolution.

    Args:
        x: Input tensor.
        filters: Number of filters produced by the transposed convolution.
        kernel_size: Size of the transposed convolution kernel.
        strides: Stride of the transposed convolution (the upsampling factor).
        padding: Padding mode used by the transposed convolution.
        hidden_activation: Optional activation applied to the output.

    Returns:
        Output tensor after applying the transposed convolution.
    """
    return Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=hidden_activation,
    )(x)


def upsample_concat_block(
    x: tf.Tensor,
    skip_connection: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (2, 2),
    strides: tuple[int, int] = (2, 2),
    padding: str = "same",
) -> tf.Tensor:
    """Upsample 'x' and concatenate it with a skip connection.

    Args:
        x: Input tensor from the previous decoder stage.
        skip_connection: Feature map to concatenate with the upsampled tensor.
        filters: Number of filters for the transposed convolution.
        kernel_size: Kernel size of the transposed convolution.
        strides: Stride of the transposed convolution (the upsampling factor).
        padding: Padding mode for the transposed convolution.

    Returns:
        The concatenated tensor, ready to be refined by further blocks.
    """
    x = conv2d_transpose(
        x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
    )
    return concatenate([x, skip_connection])


def dense_head(
    x: tf.Tensor,
    num_classes: int,
    units: list[int],
    hidden_activation: str = "relu",
    output_activation: str = "softmax",
    batch_norm: bool = False,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """Build a fully connected classification head.

    Args:
        x: Input tensor.
        num_classes: Number of output classes.
        units: Number of units for each intermediate dense layer.
        hidden_activation: Activation function applied to the hidden layers.
        output_activation: Activation function of the final classifier.
        batch_norm: Whether to apply batch normalization after each dense layer.
        dropout_rate: Dropout rate applied after each dense layer.

    Returns:
        Output tensor containing the class predictions.
    """
    x = Flatten()(x)
    for num_units in units:
        x = Dense(num_units, activation=hidden_activation)(x)

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
    x = conv2d(
        x,
        filters=3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )

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
