from omegaconf import DictConfig, OmegaConf
from tensorflow.keras.losses import CategoricalCrossentropy, Loss

LOSS_REGISTRY: dict[str, type[Loss]] = {
    "categorical-crossentropy": CategoricalCrossentropy
}


def resolve_loss(cfg_training: DictConfig) -> Loss:
    """Resolve and instantiate the configured loss function"""
    try:
        loss_cls = LOSS_REGISTRY[cfg_training.loss.name]
    except KeyError as e:
        raise ValueError(f"Unknown loss function: {cfg_training.loss.name}") from e

    params = dict(OmegaConf.to_container(cfg_training.loss.params, resolve=True))
    return loss_cls(**params)
