import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from .BaseDatasetLoader import BaseDatasetLoader


class LocalDirectoryDatasetLoader(BaseDatasetLoader):
    """Load image datasets from a directory tree structured as one subfolder per class."""

    COLOR_MODES: dict[int, str] = {
        1: "grayscale",
        3: "rgb",
        4: "rgba",
    }

    def __init__(self, cfg_dataset: DictConfig) -> None:
        super().__init__(cfg_dataset)
        self._num_classes: int | None = None
        self._class_names: list[str] | None = None

    def __str__(self) -> str:
        """Return the loader class name."""
        return self.__class__.__name__

    @property
    def num_classes(self) -> int | None:
        """Number of classes, or 'None' if load_data() has not yet been called."""
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Class names in Keras' inferred (alphabetical) order,
        or 'None' if load_data() has not yet been called."""
        return self._class_names

    def _create_dataset(
        self, directory: str, subset: str | None = None
    ) -> tf.data.Dataset:
        """Build a TensorFlow dataset from a directory via image_dataset_from_directory.

        Forwards cfg.dataset.loader.params as kwargs, overriding directory,
        labels and color_mode (derived from cfg.dataset.num_bands). When
        subset is None, validation_split is dropped so the whole directory
        is used (the test set case).

        Args:
            directory: Root directory, one subfolder per class.
            subset: "training" or "validation" to split train_dir via
                validation_split; None to load the full directory as-is.

        Returns:
            A TensorFlow dataset.

        Raises:
            ValueError: If cfg.dataset.num_bands has no matching color mode
                (see COLOR_MODES).
        """

        color_mode = self.COLOR_MODES.get(self._cfg_dataset.num_bands)

        if color_mode is None:
            raise ValueError(f"Unsupported num_bands: {self._cfg_dataset.num_bands}")

        params = dict(
            OmegaConf.to_container(self._cfg_dataset.loader.params, resolve=True)
        )

        params["directory"] = directory
        params["labels"] = "inferred"
        params["color_mode"] = color_mode

        if subset is not None:
            params["subset"] = subset
        else:
            params.pop("validation_split", None)

        return tf.keras.utils.image_dataset_from_directory(**params)

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load the training, validation and test datasets, populating num_classes and class_names.

        train_dir is split into train/validation via validation_split; test_dir is loaded as-is.

        Returns:
            A tuple containing the training, validation and test datasets.
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
