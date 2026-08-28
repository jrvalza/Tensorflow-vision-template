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

    enabled: bool
    steps: list[ParamsConfig] = field(default_factory=list)


@dataclass
class AugmentationConfig:
    """Configuration for the augmentation pipeline"""

    enabled: bool
    transforms: list[ParamsConfig] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing"""

    task: str

    dataset_name: str

    root_dir: str
    train_dir: str
    test_dir: str
    metadata_csv: str

    num_bands: int

    loader: ParamsConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig


@dataclass
class ModelConfig:
    """Configuration for model construction."""

    model_name: str
    builder: str
    blocks: list[BlockConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Configuration for training model"""

    epochs: int
    verbose: int
    loss: ParamsConfig
    optimizer: ParamsConfig
    metrics: list[str] = field(default_factory=list)
    callbacks: list[ParamsConfig] = field(default_factory=list)


@dataclass
class TestConfig:
    """Configuration for model evaluation"""

    task: str
    seed: int
    model_path: str
    # dataset: DatasetConfig


@dataclass
class TrainConfig:
    """Configuration for model training"""

    task: str
    global_seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: TestConfig
