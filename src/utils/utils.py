
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PreprocessingConfig:
    steps: List[str] = field(default_factory=list)

@dataclass
class DatasetConfig:
    dataset_name: str
    loader: str
    train_dir: str
    test_dir: str
    num_bands: int
    image_size: Tuple[int, int]
    batch_size: int
    label_mode: str
    validation_split: float
    shuffle: bool
    seed: int
    preprocessing: PreprocessingConfig

@dataclass
class ModelConfig:
    model_name: str
    type: str
    task: str
    fine_tune: bool

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
