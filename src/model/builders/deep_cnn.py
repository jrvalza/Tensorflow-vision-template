from omegaconf import DictConfig
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
)


def deep_cnn(
    cfg: DictConfig, input_shape: tuple[int, int, int], num_classes: int
) -> Model:
    """
    Build a Deep CNN classification model.

    Args:
        cfg: Project configuration.
        input_shape: Shape of the input images.
        num_classes: Number of output classes.

    Returns:
        The constructed Keras model.
    """
    batch_norm = cfg.model.batch_norm
    dropout_rate = cfg.model.dropout_rate

    inputs = Input(shape=input_shape)

    block1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    if batch_norm:
        block1 = BatchNormalization()(block1)
    block1 = Conv2D(32, (3, 3), padding="same", activation="relu")(block1)
    if batch_norm:
        block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)
    block1 = Dropout(dropout_rate / 2)(block1)

    block2 = Conv2D(64, (3, 3), padding="same", activation="relu")(block1)
    if batch_norm:
        block2 = BatchNormalization()(block2)
    block2 = Conv2D(64, (3, 3), padding="same", activation="relu")(block2)
    if batch_norm:
        block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2))(block2)
    block2 = Dropout(dropout_rate / 2)(block2)

    classifier = Flatten()(block2)
    classifier = Dense(units=512, activation="relu")(classifier)
    if batch_norm:
        classifier = BatchNormalization()(classifier)
    classifier = Dropout(dropout_rate)(classifier)

    predictions = Dense(units=num_classes, activation="softmax")(classifier)

    return Model(inputs=inputs, outputs=predictions)
