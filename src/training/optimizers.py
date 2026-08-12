from omegaconf import DictConfig, OmegaConf
from tensorflow.keras.optimizers import SGD, Adam, AdamW, Optimizer

OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}


def resolve_optimizer(cfg_training: DictConfig) -> Optimizer:
    """Resolve and instantiate the configured optimizer"""

    try:
        optimizer_cls = OPTIMIZER_REGISTRY[cfg_training.optimizer.name]
    except KeyError as e:
        raise ValueError(f"Unknown optimizer: {cfg_training.optimizer.name}") from e

    params = dict(OmegaConf.to_container(cfg_training.optimizer.params, resolve=True))
    return optimizer_cls(**params)
