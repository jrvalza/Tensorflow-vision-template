import json
import tensorflow as tf
from omegaconf import DictConfig

from .loaders.BaseDatasetLoader import BaseDatasetLoader
from .loaders.LocalDirectoryDatasetLoader import LocalDirectoryDatasetLoader
from .preprocessing.PreprocessingPipeline import PreprocessingPipeline

AUTOTUNE = tf.data.AUTOTUNE


class DatasetManager:
    """Loads and preprocesses datasets based on cfg.dataset.loader.name."""

    LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
        "from_local_directory": LocalDirectoryDatasetLoader
    }

    def __init__(self, cfg_dataset: DictConfig) -> None:
        self._cfg_dataset = cfg_dataset
        self._loader: BaseDatasetLoader | None = None
        self._preprocess_pipeline = PreprocessingPipeline(cfg_dataset)

    def __str__(self) -> str:
        """List the available dataset loaders."""
        loaders = {
            loader_type: loader_cls.__name__
            for loader_type, loader_cls in self.LOADER_REGISTRY.items()
        }
        return f"Available dataset loaders:\n{json.dumps(loaders, indent=4)}"

    @property
    def num_classes(self) -> int:
        """Number of classes in the loaded dataset.

        Raises:
            RuntimeError: If load_data() hasn't been called yet.
        """
        if self._loader is None:
            raise RuntimeError("Dataset has not been loaded.")
        return self._loader.num_classes

    @property
    def class_names(self) -> list[str]:
        """Class names of the loaded dataset, in the loader's order.

        Raises:
            RuntimeError: If load_data() hasn't been called yet.
        """
        if self._loader is None:
            raise RuntimeError("Dataset has not been loaded.")
        return self._loader.class_names

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load train/val/test datasets and apply the preprocessing pipeline.

        Returns:
            (train_ds, val_ds, test_ds), batched, preprocessed and prefetched.

        Raises:
            ValueError: If cfg.dataset.loader.name is not registered.
        """
        try:
            loader_cls = self.LOADER_REGISTRY[self._cfg_dataset.loader.name]
        except KeyError as e:
            raise ValueError(
                f"Unknown dataset loader: {self._cfg_dataset.loader.name}"
            ) from e

        self._loader = loader_cls(self._cfg_dataset)

        train_ds, val_ds, test_ds = self._loader.load_data()

        train_ds = self._preprocess_pipeline.apply(train_ds)
        val_ds = self._preprocess_pipeline.apply(val_ds)
        test_ds = self._preprocess_pipeline.apply(test_ds)

        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

        return train_ds, val_ds, test_ds
