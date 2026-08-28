from omegaconf import DictConfig, OmegaConf
from tensorflow.keras.losses import (
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    Loss,
)

LOSS_REGISTRY: dict[str, type[Loss]] = {
    "categorical_crossentropy": CategoricalCrossentropy,
    "sparse_categorical_crossentropy": SparseCategoricalCrossentropy,
}


def resolve_loss(cfg_training: DictConfig) -> Loss:
    """Instantiate the loss configured in cfg.training.loss.

    Args:
        cfg_training: 'training' section of the config. Requires loss.name
            (key in LOSS_REGISTRY) and loss.params (kwargs for the loss
            constructor).

    Returns:
        An instantiated Keras loss.

    Raises:
        ValueError: If loss.name is not registered.
    """
    try:
        loss_cls = LOSS_REGISTRY[cfg_training.loss.name]
    except KeyError as e:
        raise ValueError(f"Unknown loss function: {cfg_training.loss.name}") from e

    params = dict(OmegaConf.to_container(cfg_training.loss.params, resolve=True))
    return loss_cls(**params)
