from omegaconf import OmegaConf
from src.utils.paths import get_checkpoint_dir


def params_resolvers() -> None:
    """Register custom OmegaConf resolvers used by the application."""

    OmegaConf.register_new_resolver(
        "checkpoint_path",
        lambda: str(get_checkpoint_dir() / "best_model.keras"),
        replace=True,
    )
