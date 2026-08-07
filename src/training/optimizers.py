
from omegaconf import DictConfig
from tensorflow.keras.optimizers import SGD, Adam, Optimizer


OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "sgd": SGD,
    "adam": Adam
}

def resolve_optimizer(cfg_trainer: DictConfig) -> Optimizer:
    """Resolve and instantiate the configured optimizer"""

    try:
        optimizer_cls = OPTIMIZER_REGISTRY[cfg_trainer.optimizer]
    except KeyError as e:
        raise ValueError(f"Unknown optimizer: {cfg_trainer.optimizer}") from e
    return optimizer_cls(learning_rate=cfg_trainer.learning_rate)
