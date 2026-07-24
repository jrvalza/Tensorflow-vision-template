
import json
from omegaconf import DictConfig
from .loaders.LocalDirectoryDatasetLoader import LocalDirectoryDatasetLoader

class DatasetManager:

    LOADERS = {
        "from_local_directory": LocalDirectoryDatasetLoader
        }

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._loader = None

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
            self._loader = loader(self._cfg)
            return self._loader.load()
        
        except KeyError:
            raise ValueError(
                f"Unknown dataset loader: {self._cfg.dataset.loader}"
            )
        