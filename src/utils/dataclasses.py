
from dataclasses import dataclass, field

@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    steps: list[str] = field(default_factory=list)

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

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
    """Configuration for model creation and training."""

    model_name: str
    architecture: str
    task: str
    fine_tune: bool
    batch_norm: bool
    dropout_rate: float

@dataclass
class Config:
    """Root configuration object for the project."""

    dataset: DatasetConfig
    model: ModelConfig
