from pathlib import Path
from hydra.core.hydra_config import HydraConfig


def create_directory(path: Path) -> Path:
    """Create a directory if it does not already exist"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir() -> Path:
    """Return the Hydra output directory for the current run"""
    return Path(HydraConfig.get().runtime.output_dir)


def get_checkpoint_dir() -> Path:
    """Return the checkpoint directory, creating it if necessary"""
    return create_directory(get_output_dir() / "checkpoints")


def get_evaluation_dir() -> Path:
    """Return the evaluation directory, creating it if necessary"""
    return create_directory(get_output_dir() / "evaluation")
