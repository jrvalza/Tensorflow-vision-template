import tensorflow as tf

from src.config.schemes.dataset_scheme import DatasetConfig
from .base_dataset_loader import BaseDatasetLoader


class ImageDatasetFromDirectory(BaseDatasetLoader):
    """Loads image classification datasets from a directory structure.

    Each class is expected to be represented by a separate subdirectory.
    The training directory is split into training and validation subsets,
    while the test directory is loaded independently.
    """

    NUM_BANDS_AND_COLOR_MODES: dict[int, str] = {
        1: "grayscale",
        3: "rgb",
        4: "rgba",
    }

    def __init__(self, cfg_dataset: DatasetConfig) -> None:
        """Initialize the image dataset loader.

        Args:
            cfg_dataset: Dataset configuration containing the directories,
                loader parameters, and number of image bands.
        """
        super().__init__(cfg_dataset)
        self._num_classes: int | None = None
        self._class_names: list[str] | None = None

    @classmethod
    def num_bands_available(cls) -> list[int]:
        """Return the supported number of image bands.

        Returns:
            A list of supported band counts based on the color modes
            supported by 'image_dataset_from_directory'.
        """
        return list(cls.NUM_BANDS_AND_COLOR_MODES.keys())

    @property
    def num_classes(self) -> int | None:
        """Return the number of classes, if the dataset has been loaded."""
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Return the class names in Keras inferred alphabetical order.

        Returns:
            The class names inferred from the training directory, or
            'None' if 'load_data()' has not been called yet.
        """
        return self._class_names

    def _create_dataset(
        self, directory: str, subset: str | None = None
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from a class-based directory.

        Forwards cfg.dataset.loader.params as kwargs, overriding directory,
        labels and color_mode (derived from cfg.dataset.num_bands). When
        subset is None, validation_split is dropped so the whole directory
        is used (the test set case).

        Args:
            directory: Root directory containing one subdirectory per class.
            subset: Dataset subset to load. Supported values are
                'training', 'validation', or 'None'. 'None
                loads the entire directory.

        Returns:
             A TensorFlow dataset created from the specified directory.

        Raises:
            ValueError: If the configured number of bands is not supported.
                (see NUM_BANDS_AND_COLOR_MODES).
        """
        color_mode = self.NUM_BANDS_AND_COLOR_MODES.get(self._cfg_dataset.num_bands)

        if color_mode is None:
            raise ValueError(
                f"Unsupported num_bands: {self._cfg_dataset.num_bands}\n"
                f"Supported: {self.num_bands_available()}"
            )

        params = dict(self._cfg_dataset.loader.params)

        params["directory"] = directory
        params["labels"] = "inferred"
        params["color_mode"] = color_mode

        if subset is not None:
            params["subset"] = subset
        else:
            params.pop("validation_split", None)

        return tf.keras.utils.image_dataset_from_directory(**params)

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load the training, validation, and test datasets.

        The training directory is split into training and validation subsets
        using the configured 'validation_split'. The test directory is
        loaded independently without applying the validation split. The
        inferred class names and number of classes are stored after loading
        the training dataset.

        Returns:
            A tuple containing the training, validation, and test datasets.
        """

        train_ds = self._create_dataset(
            directory=self._cfg_dataset.train_dir, subset="training"
        )

        val_ds = self._create_dataset(
            directory=self._cfg_dataset.train_dir, subset="validation"
        )

        test_ds = self._create_dataset(directory=self._cfg_dataset.test_dir)

        self._class_names = train_ds.class_names
        self._num_classes = len(self._class_names)

        return train_ds, val_ds, test_ds
