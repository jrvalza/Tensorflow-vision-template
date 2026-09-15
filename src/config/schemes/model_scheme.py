from typing import Any
from pydantic import BaseModel, ConfigDict, ValidationInfo, Field, field_validator

from ._registry_validation import validate_in_registry


class BlocksConfig(BaseModel):
    """Configuration for specific params in keras layers."""

    model_config = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_block(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(value, info, ("model", "blocks"), "block function")


class ModelConfig(BaseModel):
    """Configuration for model construction."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    builder_name: str
    blocks: list[BlocksConfig] = Field(default_factory=list)

    @field_validator("builder_name")
    @classmethod
    def validate_builder(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(
            value, info, ("model", "builders"), "builder function"
        )
