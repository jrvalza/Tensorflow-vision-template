
import tensorflow as tf
from .BaseDatasetLoader import BaseDatasetLoader

AUTOTUNE = tf.data.AUTOTUNE

class LocalDirectoryDatasetLoader(BaseDatasetLoader):

    def __init__ (self, cfg):
        super().__init__(cfg)
        self._class_names = None
        self._num_classes = None

    def __str__(self):
        return f"LocalDirectoryDatasetLoader"

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_names(self):
        return self._class_names

    def _create_dataset(self, directory, subset=None, validation_split=None):
        color_mode = {
            1: "grayscale",
            3: "rgb",
            4: "rgba",
            }.get(self._cfg.dataset.num_bands)

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
    
    def load(self):
 
        train_dataset = self._create_dataset(
            directory=self._cfg.dataset.train_dir,
            subset='training',
            validation_split=self._cfg.dataset.validation_split
            )
                                             
        
        validation_dataset = self._create_dataset(
            directory=self._cfg.dataset.train_dir,
            subset='validation',
            validation_split=self._cfg.dataset.validation_split
            )
        
        test_dataset = self._create_dataset(
            directory=self._cfg.dataset.test_dir
            )

        self._class_names = train_dataset.class_names
        self._num_classes = len(self._class_names)

        train_dataset = train_dataset.prefetch(AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(AUTOTUNE)
        test_dataset = test_dataset.prefetch(AUTOTUNE)

        return (train_dataset, validation_dataset, test_dataset)
        