import rasterio
import pandas as pd
import tensorflow as tf
from collections.abc import Callable
from sklearn.model_selection import train_test_split

from .base_dataset_loader import BaseDatasetLoader
from src.config.schemes.dataset_scheme import DatasetConfig


def _read_raster(path: str) -> tf.Tensor:
    """Read a raster file into a TensorFlow tensor.

    Rasterio returns raster data with shape (bands, H, W). The resulting
    tensor is transposed to (H, W, bands) to match TensorFlow's expected
    image layout.

    Args:
        path: Path to the raster file.

    Returns:
        Tensor containing the raster data with shape (H, W, bands).
    """
    with rasterio.open(path) as src:
        array = src.read()
    tensor = tf.convert_to_tensor(array)
    return tf.transpose(tensor, [1, 2, 0])


def _tf_py_function(full_path: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    """Read a raster file using TensorFlow's Python callback mechanism.

    This function wraps the Rasterio-based raster reader with
    'tf.py_function' so that raster files can be loaded inside a
    'tf.data' pipeline.

    Args:
        full_path: Tensor containing the path to the raster file.
        dtype: TensorFlow dtype of the returned tensor.

    Returns:
        Tensor containing the raster data.
    """
    return tf.py_function(
        lambda path: _read_raster(path.numpy().decode("utf-8")),
        [full_path],
        dtype,
    )


class CSVDatasetLoader(BaseDatasetLoader):
    """Loads image-target pairs from a CSV metadata file.

    The CSV file must contain an image path and a corresponding label or mask
    path for each sample. Training and test samples are identified by the
    'Train/' and 'Test/' path prefixes, respectively. The training samples
    are further split into training and validation subsets.
    """

    COLUMN_NAMES: list[str] = ["image_path", "target"]

    def __init__(self, cfg_dataset: DatasetConfig) -> None:
        """Initialize the CSV dataset loader.

        Args:
            cfg_dataset: Dataset configuration containing the CSV metadata path,
                dataset root directory, task, image properties, and loader
                parameters.
        """
        super().__init__(cfg_dataset)
        self._num_classes: int | None = None
        self._class_names: list[str] | None = None
        self._label_table: tf.lookup.StaticHashTable | None = None
        self._label_loaders: dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
            "classification": self._load_classification_label,
            "segmentation": self._load_segmentation_mask,
        }

    @property
    def num_classes(self) -> int | None:
        """Return the number of classes, if the dataset has been loaded."""
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Return the dataset class names, if the dataset has been loaded."""
        return self._class_names

    def _split_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read the metadata CSV and split samples into train, validation, and test sets.

        Training and test samples are identified by the 'Train/' and 'Test/'
        prefixes in the image paths. The training samples are further split into
        training and validation subsets using the configured validation split
        and random seed.

        Returns:
            A tuple containing the training, validation, and test dataframes.

        Raises:
            ValueError: If the CSV file does not contain samples for both the
                'Train/' and 'Test/' subsets.
        """
        df = pd.read_csv(
            self._cfg_dataset.metadata_csv,
            sep=None,
            engine="python",
            header=0,
            names=self.COLUMN_NAMES,
        )

        df_train = df[df[self.COLUMN_NAMES[0]].str.startswith("Train/")]
        df_test = df[df[self.COLUMN_NAMES[0]].str.startswith("Test/")]

        if df_train.empty or df_test.empty:
            raise ValueError(
                "The 'metadata_csv' file must contain rows with the prefix 'Train/' and 'Test/'."
            )

        df_train, df_val = train_test_split(
            df_train,
            test_size=self._cfg_dataset.loader.params.validation_split,
            random_state=self._cfg_dataset.loader.params.seed,
        )

        return df_train, df_val, df_test

    def _resolve_classes(self, df_train: pd.DataFrame) -> None:
        """Resolve class names and label encoding for the configured task.

        For classification, class names are inferred from the training targets
        and a class-name-to-index lookup table is created.
        For segmentation, class names are obtained from the configuration or
        generated from 'num_classes'.

        Args:
            df_train: DataFrame containing the training targets.

        Raises:
            ValueError: If segmentation is configured without class names.
        """
        if self._cfg_dataset.task == "classification":
            targets_as_str = df_train[self.COLUMN_NAMES[1]].astype(str)
            self._class_names = sorted(targets_as_str.unique().tolist())

            self._label_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(self._class_names), tf.range(len(self._class_names))
                ),
                default_value=-1,
            )

        if self._cfg_dataset.task == "segmentation":
            self._class_names = list(self._cfg_dataset.class_names.values())
        self._num_classes = len(self._class_names)

    def _load_classification_label(self, target: tf.Tensor) -> tf.Tensor:
        """Convert a class name into its encoded label representation.

        Class names are mapped to integer indices using the lookup table created
        during class resolution. If 'label_mode' is 'categorical', the
        integer index is converted to a one-hot encoded tensor.

        Args:
            target: Tensor containing the class name.

        Returns:
            A tensor containing either the integer class index or its one-hot
            encoded representation.
        """
        if target.dtype != tf.string:
            target = tf.strings.as_string(target)

        index = self._label_table.lookup(target)

        if self._cfg_dataset.loader.params.label_mode == "categorical":
            return tf.one_hot(index, depth=self._num_classes)
        return index

    def _load_segmentation_mask(self, target: tf.Tensor) -> tf.Tensor:
        """Load a segmentation mask from the target path.

        The target path is resolved relative to the configured dataset root
        directory. The resulting mask is returned with shape (H, W, 1).

        Args:
            target: Tensor containing the relative path to the segmentation mask.

        Returns:
            Tensor containing the per-pixel class indices with shape
            (H, W, 1).
        """
        mask_full_path = tf.strings.join([self._cfg_dataset.root_dir, "/", target])
        mask = _tf_py_function(
            full_path=mask_full_path, dtype=tf.as_dtype(self._cfg_dataset.image_dtype)
        )
        mask.set_shape([None, None, 1])
        return mask

    def _load_pair(
        self, image_path: tf.Tensor, target: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Load an image and its corresponding target.

        The image path is resolved relative to the configured dataset root
        directory. The target is loaded according to the configured task,
        either as a classification label or a segmentation mask.

        Args:
            image_path: Tensor containing the relative path to the image.
            target: Tensor containing the corresponding class label or mask path.

        Returns:
            A tuple containing the image tensor and its corresponding target.
        """
        image_full_path = tf.strings.join([self._cfg_dataset.root_dir, "/", image_path])
        image = _tf_py_function(
            full_path=image_full_path, dtype=tf.as_dtype(self._cfg_dataset.image_dtype)
        )
        image.set_shape([None, None, self._cfg_dataset.num_bands])
        label = self._label_loaders[self._cfg_dataset.task](target)
        return image, label

    def _make_dataset(
        self, df: pd.DataFrame, training: bool = False
    ) -> tf.data.Dataset:
        """Create a batched TensorFlow dataset from a dataframe.

        Samples are optionally shuffled when creating the training dataset,
        according to the configured 'shuffle' parameter. Images and their
        corresponding targets are then loaded in parallel and batched using the
        configured batch size.

        Args:
            df: DataFrame containing image paths and target information.
            training: Whether the dataset is used for training and should be
                eligible for shuffling.

        Returns:
            A batched TensorFlow dataset containing images and their targets.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (df[self.COLUMN_NAMES[0]].to_numpy(), df[self.COLUMN_NAMES[1]].to_numpy())
        )

        if training and self._cfg_dataset.loader.params.shuffle:
            dataset = dataset.shuffle(
                len(df), seed=self._cfg_dataset.loader.params.seed
            )
        dataset = dataset.map(self._load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(self._cfg_dataset.loader.params.batch_size)

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare the train, validation, and test datasets.

        The metadata CSV is split into training, validation, and test samples.
        Class information is resolved from the training data or configuration,
        and each split is converted into a batched TensorFlow dataset.

        Returns:
            A tuple containing the batched training, validation, and test
            datasets.
        """
        df_train, df_val, df_test = self._split_dataframe()
        self._resolve_classes(df_train)

        #####CHANGE MESSAGE PRINT FOR LOGGING
        message = (
            f"Found {len(df_train) + len(df_val)} files belonging to {self._num_classes} classes.\n"
            f" - Using {len(df_train)} files for training and {len(df_val)} files for validation "
            f"({self._cfg_dataset.loader.params.validation_split * 100}% of Train dataset).\n\n"
            f"Found {len(df_test)} files belonging to {self._num_classes} classes.\n"
            f" - Using {len(df_test)} files for test."
        )
        print(message)

        train_ds = self._make_dataset(df_train, training=True)
        val_ds = self._make_dataset(df_val)
        test_ds = self._make_dataset(df_test)

        return train_ds, val_ds, test_ds
