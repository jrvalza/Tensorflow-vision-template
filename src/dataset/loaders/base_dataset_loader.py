from abc import ABC, abstractmethod

import tensorflow as tf
from omegaconf import DictConfig


class BaseDatasetLoader(ABC):
    """Base interface for dataset loaders."""

    def __init__(self, cfg_dataset: DictConfig) -> None:
        self._cfg_dataset = cfg_dataset

    @property
    @abstractmethod
    def num_classes(self) -> int | None:
        """Number of classes in the dataset, or None if not loaded yet."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> list[str] | None:
        """Dataset class names, or None if not loaded yet."""
        pass

    @abstractmethod
    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load the training, validation and test datasets.

        Returns:
            (train_ds, val_ds, test_ds).
        """
        pass
