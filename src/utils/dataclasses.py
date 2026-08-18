from typing import Any
from dataclasses import dataclass, field


@dataclass
class ParamsConfig:
    """Configuration for specific params in loaders, losses, optimizers and callbacks"""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockConfig:
    """Configuration for specific params in keras layers"""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline"""

    steps: list[ParamsConfig] = field(default_factory=list)


@dataclass
class AugmentationConfig:
    """Configuration for the augmentation pipeline"""

    enabled: bool
    transforms: list[ParamsConfig] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing"""

    dataset_name: str
    num_bands: int
    train_dir: str
    test_dir: str

    loader: ParamsConfig

    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig


@dataclass
class ModelConfig:
    """Configuration for model construction."""

    model_name: str
    builder: str
    task: str
    blocks: list[BlockConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Configuration for training model"""

    epochs: int
    loss: ParamsConfig
    optimizer: ParamsConfig
    metrics: list[str] = field(default_factory=list)
    callbacks: list[ParamsConfig] = field(default_factory=list)


@dataclass
class TestConfig:
    """Configuration for model evaluation"""

    dataset: DatasetConfig
    model_path: str


@dataclass
class TrainConfig:
    """Configuration for model training"""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
