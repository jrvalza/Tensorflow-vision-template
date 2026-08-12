from omegaconf import DictConfig, OmegaConf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from src.utils.paths import get_checkpoint_dir

CALLBACK_REGISTRY: dict[str, type[Callback]] = {
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
}


def resolve_callbacks(cfg_training: DictConfig) -> list[Callback]:
    """Instantiate the callbacks configured in cfg.training.callbacks.

    Args:
        cfg_training: 'training' section of the config. callbacks must be a
            list of {name, params} entries, name being a key in
            CALLBACK_REGISTRY. For ModelCheckpoint, filepath is injected
            automatically if not set.

    Returns:
        The instantiated Keras callbacks, in configured order.

    Raises:
        ValueError: If a callback name is not registered.
    """
    callbacks = []
    for cfg_callback in cfg_training.callbacks:

        try:
            callback_cls = CALLBACK_REGISTRY[cfg_callback.name]
        except KeyError as e:
            raise ValueError(f"Unknown callback: {cfg_callback.name}") from e

        params = dict(OmegaConf.to_container(cfg_callback.params, resolve=True))

        if callback_cls is ModelCheckpoint:
            params["filepath"] = str(get_checkpoint_dir() / "best_model.keras")

        callbacks.append(callback_cls(**params))

    return callbacks
