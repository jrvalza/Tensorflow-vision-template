
from omegaconf import DictConfig
from tensorflow.keras.optimizers import (
    SGD, 
    Adam,
    AdamW, 
    Optimizer
)


OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW
}

def resolve_optimizer(cfg_trainer: DictConfig) -> Optimizer:
    """Resolve and instantiate the configured optimizer"""

    try:
        optimizer_cls = OPTIMIZER_REGISTRY[cfg_trainer.optimizer.name]
    except KeyError as e:
        raise ValueError(f"Unknown optimizer: {cfg_trainer.optimizer.name}") from e
    return optimizer_cls(**cfg_trainer.optimizer.params)
