
from abc import ABC, abstractmethod

import tensorflow as tf
from omegaconf import DictConfig


class BaseDatasetLoader(ABC):
    """Base interface for dataset loaders"""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Return the dataset class names."""
        pass

    @abstractmethod
    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load the training, validation and test datasets."""
        pass
