from omegaconf import DictConfig
from tensorflow.keras.losses import CategoricalCrossentropy, Loss

LOSS_REGISTRY: dict[str, type[Loss]] = {
    "categorical-crossentropy": CategoricalCrossentropy
}


def resolve_loss(cfg_trainer: DictConfig) -> Loss:
    """Resolve and instantiate the configured loss function"""
    try:
        loss_cls = LOSS_REGISTRY[cfg_trainer.loss.name]
    except KeyError as e:
        raise ValueError(f"Unknown loss function: {cfg_trainer.loss.name}") from e
    return loss_cls(**cfg_trainer.loss.params)
