"""Central registry of dataset loaders, preprocessing steps, and augmentations.

To add a new loader, preprocessing step, or augmentation, implement it in
'src/dataset' and register it here under a unique string key. That key is
what users reference from YAML config, and is validated against these
registries by the corresponding configuration schemas.

"""

from collections.abc import Callable
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    Layer,
)
from src.dataset.loaders.csv_dataset_loader import CSVDatasetLoader
from src.dataset.loaders.base_dataset_loader import BaseDatasetLoader
from src.dataset.loaders.image_dataset_from_directory import ImageDatasetFromDirectory
from src.dataset.preprocessing.preprocessing_steps import (
    pixel_value_normalization,
    resize_image,
)
from src.dataset.preprocessing.preprocessing_pipeline import Step

LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    "from_local_directory": ImageDatasetFromDirectory,
    "from_csv_metadata": CSVDatasetLoader,
}


PREPROCESSING_STEPS_REGISTRY: dict[str, Callable[..., Step]] = {
    "pixel_value_normalization": pixel_value_normalization,
    "resize_image": resize_image,
}

AUGMENTATION_STEPS_REGISTRY: dict[str, Callable[..., Layer]] = {
    "random_flip": RandomFlip,
    "random_rotation": RandomRotation,
    "random_zoom": RandomZoom,
    "random_translation": RandomTranslation,
}
