from pathlib import Path
from typing import Any, Annotated
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    Field,
    field_validator,
    model_validator,
)

from ._registry_validation import validate_in_registry
from ..literals import literal_cv_tasks, literal_label_modes, literal_image_dtypes


class LoaderParamsConfig(BaseModel):
    """Configuration parameters for dataset loading."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(gt=0)
    image_size: tuple[Annotated[int, Field(gt=8)], Annotated[int, Field(gt=8)]]
    validation_split: float = Field(gt=0.0, lt=1.0)
    label_mode: literal_label_modes
    seed: int = Field(ge=0)
    shuffle: bool


class LoaderConfig(BaseModel):
    """Configuration for a dataset loader."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: LoaderParamsConfig

    @field_validator("name")
    @classmethod
    def validate_loader(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(
            value, info, ("dataset", "loaders"), "loader function"
        )


class PreprocessingStepConfig(BaseModel):
    """Configuration for specific params in preprocessing steps"""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_step(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(
            value,
            info,
            ("dataset", "preprocessing_steps"),
            "preprocessing step function",
        )


class PreprocessingConfig(BaseModel):
    """Configuration for the preprocessing pipeline"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    steps: list[PreprocessingStepConfig] = Field(default_factory=list)


class AugmentationStepConfig(BaseModel):
    """Configuration for specific params in augmentation steps"""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_step(cls, value: str, info: ValidationInfo) -> str:
        return validate_in_registry(
            value, info, ("dataset", "augmentation_steps"), "augmentation step function"
        )


class AugmentationConfig(BaseModel):
    """Configuration for the augmentation pipeline"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    transforms: list[AugmentationStepConfig] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and preprocessing"""

    model_config = ConfigDict(extra="forbid")

    task: literal_cv_tasks

    dataset_name: str

    root_dir: str
    train_dir: str
    test_dir: str
    metadata_csv: str

    num_bands: int = Field(gt=0)
    image_dtype: literal_image_dtypes
    class_names: dict[int, str] | None = None

    loader: LoaderConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig

    @model_validator(mode="after")
    def check_paths_for_loader(self) -> "DatasetConfig":
        """Validate only the filesystem paths required by the selected loader."""
        if self.loader.name == "from_local_directory":
            for field_name in ("train_dir", "test_dir"):
                value = getattr(self, field_name)
                if not Path(value).is_dir():
                    raise ValueError(
                        f"'{field_name}' does not exist or is not a directory: {value}"
                    )
        elif self.loader.name == "from_csv_metadata":
            if not Path(self.root_dir).is_dir():
                raise ValueError(
                    f"'root_dir' does not exist or is not a directory: {self.root_dir}"
                )
            if not Path(self.metadata_csv).is_file():
                raise ValueError(
                    f"'metadata_csv' does not exist or is not a file: {self.metadata_csv}"
                )
        return self

    @model_validator(mode="after")
    def require_class_names_for_segmentation(self) -> "DatasetConfig":
        if self.task == "segmentation" and not self.class_names:
            raise ValueError(
                "'class_names' must be provided (as {index: name}) when task='segmentation'."
            )
        return self
