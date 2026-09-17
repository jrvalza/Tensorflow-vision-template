from typing import Any
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    Field,
    field_validator,
    model_validator,
)

from ._registry_validation import validate_in_registry


class BlocksConfig(BaseModel):
    """Configuration for specific params in keras layers."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    skip_connection_from: str | None = None
    inject_num_classes_as: str | None = None
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

    @model_validator(mode="after")
    def check_block_graph(self) -> "ModelConfig":
        """Ensure block names are unique and 'skip_connection_from' references a prior block.

        Raises:
            ValueError: If a block name is repeated, or if 'skip_connection_from' references a block
                name that is not defined earlier in the sequence.
        """
        block_names: set[str] = set()
        for block in self.blocks:
            if block.name in block_names:
                raise ValueError(f"Duplicate block name: '{block.name}'")
            if (
                block.skip_connection_from is not None
                and block.skip_connection_from not in block_names
            ):
                raise ValueError(
                    f"Block '{block.name}' has skip_connection_from='{block.skip_connection_from}', "
                    "which is not a previously defined block name."
                )
            block_names.add(block.name)
        return self
