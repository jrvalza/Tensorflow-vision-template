import tensorflow as tf
from abc import ABC, abstractmethod

from src.config.schemes.dataset_scheme import DatasetConfig


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Defines the interface that concrete dataset loaders must implement to
    provide dataset metadata and load the training, validation, and test
    splits.
    """

    def __init__(self, cfg_dataset: DatasetConfig) -> None:
        """Initialize the dataset loader.
        Args:
            cfg_dataset: Dataset configuration used by the concrete loader.
        """
        self._cfg_dataset = cfg_dataset

    @property
    @abstractmethod
    def num_classes(self) -> int | None:
        """Return the number of classes in the dataset, if available."""
        ...

    @property
    @abstractmethod
    def class_names(self) -> list[str] | None:
        """Return the dataset class names, if available."""
        ...

    @abstractmethod
    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load the training, validation, and test datasets.

        Returns:
            A tuple containing the training, validation, and test datasets.
        """
        ...
