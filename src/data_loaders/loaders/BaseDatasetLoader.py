
from abc import ABC, abstractmethod

class BaseDatasetLoader(ABC):

    def __init__(self, cfg):
        self._cfg = cfg

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    @abstractmethod
    def class_names(self):
        pass

    @abstractmethod
    def load(self):
        """
        return train_ds, val_ds, test_ds
        """
        pass
