
import json

import tensorflow as tf
from omegaconf import DictConfig
from .BaseDatasetLoader import BaseDatasetLoader

from src.utils.paths import get_checkpoint_dir


class LocalDirectoryDatasetLoader(BaseDatasetLoader):
    """Load image datasets from a local directory."""
    
    COLOR_MODES: dict[int, str] = {
        1: "grayscale",
        3: "rgb",
        4: "rgba"
    }

    def __init__ (self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._num_classes: int | None = None
        self._class_names: list[str] | None = None
        
    def __str__(self) -> str:
        """Return the loader name."""
        return self.__class__.__name__
    
    @property
    def num_classes(self) -> int | None:
        """Return the number of classes in the dataset."""
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Return the dataset class names."""
        return self._class_names

    def _create_dataset(self, directory: str, subset: str | None = None, validation_split: float | None = None) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from a directory.

        Args:
            directory: Path to the image directory.
            subset: Dataset subset ("training" or "validation").
            validation_split: Fraction of data reserved for validation.

        Returns:
            A TensorFlow dataset.
        """

        color_mode = self.COLOR_MODES.get(self._cfg.dataset.num_bands)

        if color_mode is None:
            raise ValueError(
                f"Unsupported num_bands: {self._cfg.dataset.num_bands}"
            )

        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            color_mode=color_mode,
            label_mode=self._cfg.dataset.label_mode,
            image_size=tuple(self._cfg.dataset.image_size),
            batch_size=self._cfg.dataset.batch_size,
            validation_split=validation_split,
            subset=subset,
            shuffle=self._cfg.dataset.shuffle,
            seed=self._cfg.dataset.seed
            )
    
    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load the training, validation and test datasets from local directories.

        Returns:
            A tuple containing the training, validation and test datasets.
        """
        
        train_ds = self._create_dataset(
            directory=self._cfg.dataset.train_dir,
            subset='training',
            validation_split=self._cfg.dataset.validation_split
            )
                                             
        val_ds = self._create_dataset(
            directory=self._cfg.dataset.train_dir,
            subset='validation',
            validation_split=self._cfg.dataset.validation_split
            )
        
        test_ds = self._create_dataset(
            directory=self._cfg.dataset.test_dir
            )

        self._class_names = train_ds.class_names
        self._num_classes = len(self._class_names)
     
        checkpoint_dir = get_checkpoint_dir()
        class_names_path = str(checkpoint_dir / "class_names.json")

        with open(class_names_path, "w") as f:
            json.dump(self.class_names, f, indent=2)

        return train_ds, val_ds, test_ds
        