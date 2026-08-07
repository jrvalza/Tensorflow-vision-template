
from typing import Any
from dataclasses import dataclass, field

@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline"""

    steps: list[str] = field(default_factory=list)

@dataclass
class ParamsConfig:
    """Configuration for specific params"""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing"""

    dataset_name: str
    loader: str

    train_dir: str
    test_dir: str

    validation_split: float

    num_bands: int
    image_size: tuple[int, int]
    batch_size: int
    label_mode: str
    
    shuffle: bool
    seed: int

    preprocessing: PreprocessingConfig

@dataclass
class ModelConfig:
    """Configuration for model creation and training"""

    model_name: str
    architecture: str
    task: str
    fine_tune: bool
    batch_norm: bool
    dropout_rate: float
    
@dataclass
class TrainingConfig:
    """Configuration for training model"""
    
    loss: str
    epochs: int
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
    trainer: TrainingConfig
