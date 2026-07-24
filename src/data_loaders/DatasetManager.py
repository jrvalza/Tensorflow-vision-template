
import json
import tensorflow as tf
from omegaconf import DictConfig

from .loaders.LocalDirectoryDatasetLoader import LocalDirectoryDatasetLoader
from .preprocessing.PreprocessingPipeline import PreprocessingPipeline

AUTOTUNE = tf.data.AUTOTUNE

class DatasetManager:

    LOADERS = {
        "from_local_directory": LocalDirectoryDatasetLoader
        }

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._loader = None
        self._preprocess_pipeline = PreprocessingPipeline(cfg)

    def __str__(self):
        loaders = {loader_type: loader_cls.__name__ for loader_type, loader_cls in self.LOADERS.items()}
        return f"Available dataset loaders:\n{json.dumps(loaders, indent=4)}"

    @property
    def num_classes(self):
        return self._loader.num_classes

    @property
    def class_names(self):
        return self._loader.class_names

    def load_data(self):
        try:
            loader = self.LOADERS[self._cfg.dataset.loader]    
        except KeyError:
            raise ValueError(
                f"Unknown dataset loader: {self._cfg.dataset.loader}"
            )
        
        self._loader = loader(self._cfg)
        print(f"Usando loader: {self._loader}")
        
        train_ds, val_ds, test_ds = self._loader.load()

        train_ds = self._preprocess_pipeline.apply(train_ds)
        val_ds = self._preprocess_pipeline.apply(val_ds)
        test_ds = self._preprocess_pipeline.apply(test_ds)

        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

        return train_ds, val_ds, test_ds
    