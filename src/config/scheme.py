from pydantic import BaseModel, ConfigDict, Field, model_validator

from .literals import literal_cv_tasks
from .schemes.model_scheme import ModelConfig
from .schemes.dataset_scheme import DatasetConfig
from .schemes.training_scheme import TrainingConfig


class TestConfig(BaseModel):
    """Configuration for model evaluation"""

    model_config = ConfigDict(extra="forbid")

    task: literal_cv_tasks
    seed: int = Field(ge=0)
    model_path: str
    # dataset: DatasetConfig


class TrainConfig(BaseModel):
    """Configuration for model training"""

    model_config = ConfigDict(extra="forbid")

    task: literal_cv_tasks
    global_seed: int = Field(ge=0)
    reproducibility: bool
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig

    @model_validator(mode="after")
    def check_task_consistency(self) -> "TrainConfig":
        if self.dataset.task != self.task:
            raise ValueError(
                f"Inconsistent 'task' across config sections: "
                f"top-level={self.task!r}, dataset={self.dataset.task!r}"
            )
        return self
