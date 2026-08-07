
import json
import tensorflow as tf
from omegaconf import DictConfig

from .loaders.BaseDatasetLoader import BaseDatasetLoader
from .loaders.LocalDirectoryDatasetLoader import LocalDirectoryDatasetLoader
from .preprocessing.PreprocessingPipeline import PreprocessingPipeline


AUTOTUNE = tf.data.AUTOTUNE


class DatasetManager:
    """Manager class for dataset loading, preprocessing and access to dataset metadata."""

    LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
        "from_local_directory": LocalDirectoryDatasetLoader
    }

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._loader: BaseDatasetLoader | None = None
        self._preprocess_pipeline = PreprocessingPipeline(cfg)

    def __str__(self) -> str:
        """Return the available dataset loaders."""
        loaders = {loader_type: loader_cls.__name__ for loader_type, loader_cls in self.LOADER_REGISTRY.items()}
        return f"Available dataset loaders:\n{json.dumps(loaders, indent=4)}"

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the loaded dataset."""
        if self._loader is None:
            raise RuntimeError("Dataset has not been loaded.")
        return self._loader.num_classes

    @property
    def class_names(self) -> list[str]:
        """Return the class names of the loaded dataset."""
        if self._loader is None:
            raise RuntimeError("Dataset has not been loaded.")
        return self._loader.class_names

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load, preprocess and optimize the datasets."""
            
        try:
            loader_cls = self.LOADER_REGISTRY[self._cfg.dataset.loader]    
        except KeyError as e:
            raise ValueError(f"Unknown dataset loader: {self._cfg.dataset.loader}") from e
        
        self._loader = loader_cls(self._cfg)
        
        train_ds, val_ds, test_ds = self._loader.load_data()

        train_ds = self._preprocess_pipeline.apply(train_ds)
        val_ds = self._preprocess_pipeline.apply(val_ds)
        test_ds = self._preprocess_pipeline.apply(test_ds)

        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

        return train_ds, val_ds, test_ds
    