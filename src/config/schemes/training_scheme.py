from typing import Any
from pydantic import BaseModel, ConfigDict, ValidationInfo, Field, field_validator

from ._registry_validation import validate_in_registry
from ..literals import literal_metrics, literal_verbose_fit


class LossConfig(BaseModel):
    """Configuration for a model loss function."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_loss(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(
            value, info, ("training", "losses"), "loss function"
        )


class OptimizerParamsConfig(BaseModel):
    """Parameters used to configure a model optimizer."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(gt=0.0)


class OptimizerConfig(BaseModel):
    """Configuration for a model optimizer."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: OptimizerParamsConfig

    @field_validator("name")
    @classmethod
    def validate_optimizer(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(
            value, info, ("training", "optimizers"), "optimizer function"
        )


class CallbackConfig(BaseModel):
    """Configuration for a training callback."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_callback(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(
            value, info, ("training", "callbacks"), "callback function"
        )


class TrainingConfig(BaseModel):
    """Configuration for training model"""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(gt=0)
    metrics: literal_metrics
    verbose: literal_verbose_fit
    loss: LossConfig
    optimizer: OptimizerConfig

    callbacks: list[CallbackConfig] = Field(default_factory=list)
