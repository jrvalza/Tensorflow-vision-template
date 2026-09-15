from functools import partial
from collections.abc import Callable

from .base_pipeline import BasePipeline, Step
from src.config.schemes.dataset_scheme import DatasetConfig
from src.config.schemes.dataset_scheme import PreprocessingStepConfig


class PreprocessingPipeline(BasePipeline):
    """Applies the preprocessing steps configured for the dataset.

    Each configured step is resolved using the provided registry and its
    parameters are bound to the corresponding callable before execution.
    """

    def __init__(
        self,
        cfg_dataset: DatasetConfig,
        steps_registry: dict[str, Callable[..., Step]],
    ) -> None:
        """Initialize the preprocessing pipeline.

        Args:
            cfg_dataset: Configuration containing the preprocessing
                steps and their parameters.
            steps_registry: Registry mapping step names to preprocessing
                callables.
        """
        self._steps_registry = steps_registry
        super().__init__(cfg_dataset)

    def _config_entries(self) -> list[PreprocessingStepConfig]:
        return self._cfg_dataset.preprocessing.steps

    def _resolve_entry(self, entry: PreprocessingStepConfig) -> Step:
        step_fn = self._steps_registry[entry.name]
        return partial(step_fn, **entry.params)
