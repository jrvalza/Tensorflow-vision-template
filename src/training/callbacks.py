from omegaconf import DictConfig
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from src.utils.paths import get_checkpoint_dir

CALLBACK_REGISTRY: dict[str, type[Callback]] = {
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
}


def resolve_callbacks(cfg_trainer: DictConfig) -> list[Callback]:
    """Resolve and instantiate the configured callback"""
    callbacks = []
    for cfg_callback in cfg_trainer.callbacks:

        try:
            callback_cls = CALLBACK_REGISTRY[cfg_callback.name]
        except KeyError as e:
            raise ValueError(f"Unknown callback: {cfg_callback.name}") from e

        params = dict(cfg_callback.params)

        if callback_cls is ModelCheckpoint:
            params["filepath"] = str(get_checkpoint_dir() / "best_model.keras")

        callbacks.append(callback_cls(**params))

    return callbacks
