import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import rasterio
from collections.abc import Callable
from omegaconf import OmegaConf, DictConfig

from .base_dataset_loader import BaseDatasetLoader


def _read_raster(path: str) -> tf.Tensor:
    """Read a raster file and return it as an (H, W, bands)tensor.

    Args:
        path: Path to the raster file.

    Returns:
        A tensor containing the raster data with shape (H, W, bands).
    """
    with rasterio.open(path) as src:
        array = src.read()
    tensor = tf.convert_to_tensor(array)
    return tf.transpose(tensor, [1, 2, 0])


def _tf_py_function(full_path: str, dtype: tf.DType):
    return tf.py_function(
        lambda path: _read_raster(path.numpy().decode("utf-8")),
        [full_path],
        dtype,
    )


class CSVDatasetLoader(BaseDatasetLoader):
    """Loads pairs of (image, label) or (image, mask) listed in a single CSV metadata file.

    The CSV file must contain two columns: an image path and a label or mask path,
    both relative to cfg_dataset.root_dir. Rows are assigned to training or test sets
    based on whether the image path starts with "Train/" or "Test/"; a fraction of the
    training rows is reserved for validation. The strategy used to read the second column
    is determined by cfg_dataset.task.
    """

    COLUMN_NAMES: list[str] = ["image_path", "target"]

    def __init__(self, cfg_dataset: DictConfig) -> None:
        super().__init__(cfg_dataset)
        self._num_classes: int | None = None
        self._class_names: list[str] | None = None
        self._label_table: tf.lookup.StaticHashTable | None = None
        self._label_loaders: dict[str, Callable] = {
            "classification": self._load_classification_label,
            "segmentation": self._load_segmentation_mask,
        }

    def __str__(self) -> str:
        """Return the loader class name."""
        return self.__class__.__name__

    @property
    def num_classes(self) -> int | None:
        """Number of classes, set after load_data() runs; None before that."""
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Class names; None until load_data() runs."""
        return self._class_names

    def _split_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read the CSV file and split rows into train, validation and test sets.

        Returns:
            A tuple containing the train, validation and test dataframes.

        Raises:
            ValueError: If no rows in the CSV file matches the prefixes
            "Train/" or "Test/".
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

        message = f"Found {len(df_train)+len(df_val)} files belonging to {self._num_classes} classes. \n Using {len(df_train)} files for training and {len(df_val)} files for validation ({self._cfg_dataset.loader.params.validation_split*100}% of Train dataset).\nFound {len(df_test)} files belonging to {self._num_classes} classes. \n Using {len(df_test)} files for test."
        print(message)
        return df_train, df_val, df_test

    def _resolve_classes(self, df_train: pd.DataFrame) -> None:
        """Resolve class names and indices for the configured task.

        For classification, class names are inferred from the training targets
        and a class-name-to-index lookup table is created. For segmentation,
        class names are obtained from the configuration or generated from
        'num_classes'.

        Args:
            df_train: DataFrame containing the training targets.
        """
        if self._cfg_dataset.task == "classification":

            df_train[self.COLUMN_NAMES[1]] = df_train[self.COLUMN_NAMES[1]].apply(
                lambda target: (
                    str(target)
                    if self._cfg_dataset.loader.params.label_mode == "categorical"
                    else int(target)
                )
            )

            self._class_names = sorted(df_train[self.COLUMN_NAMES[1]].unique().tolist())

            self._label_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(self._class_names), tf.range(len(self._class_names))
                ),
                default_value=-1,
            )

        if self._cfg_dataset.task == "segmentation":
            if self._cfg_dataset.class_names is None:
                raise ValueError(
                    "The class names must be specified as a dictionary under the 'dataset.class_names' key.\n"
                    f"Currently the value of class_names is: {self._cfg_dataset.class_names}"
                )

            self._class_names = list(
                OmegaConf.to_container(
                    self._cfg_dataset.class_names, resolve=True
                ).values()
            )

        self._num_classes = len(self._class_names)

    def _load_classification_label(self, target: tf.Tensor) -> tf.Tensor:
        """Convert class labels to integer indices or one-hot encodings.

        The output representation is determined by 'label_mode':
        'categorical' returns one-hot encoded labels, while other modes
        return integer class indices.

        Args:
            target: Tensor containing class-name labels.

        Returns:
            A tensor containing either integer class indices or one-hot encoded
            class labels.
        """
        target = tf.cast(target, self._label_table.key_dtype)
        index = self._label_table.lookup(target)

        if self._cfg_dataset.loader.params.label_mode == "categorical":
            return tf.one_hot(index, depth=self._num_classes)
        return index

    def _load_segmentation_mask(self, target: tf.Tensor) -> tf.Tensor:
        """Loads a segmentation mask raster from the target path.
        Each pixel stores an integer class index.

        Args:
            target: Tensor containing the path to the mask image

        Returns:
            A tensor containing the per-pixel class indices with shape (H, W, 1)
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

        Args:
            image_path: Tensor containing the image path.
            target: Tensor containing the target information.

        Returns:
            A tuple of tensors containing the loaded image and its corresponding label.
        """
        image_full_path = tf.strings.join([self._cfg_dataset.root_dir, "/", image_path])
        image = _tf_py_function(
            full_path=image_full_path, dtype=tf.as_dtype(self._cfg_dataset.image_dtype)
        )
        image.set_shape([None, None, self._cfg_dataset.num_bands])
        label = self._label_loaders[self._cfg_dataset.task](target)
        return image, label

    def _make_dataset(self, df: pd.DataFrame, training: bool = None) -> tf.data.Dataset:
        """Create a TensorFlow dataset from the input samples.
        Args:
            df: DataFrame containing image paths and target information.
            training: in case the dataset is used for training and needs to be shuffled.

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
        """Load and prepare the train, validation, and test datasets from the CSV metadata file.

        Returns:
            A tuple containing the batched train, validation, and test datasets.
        """
        df_train, df_val, df_test = self._split_dataframe()
        self._resolve_classes(df_train)

        train_ds = self._make_dataset(df_train, training=True)
        val_ds = self._make_dataset(df_val)
        test_ds = self._make_dataset(df_test)

        return train_ds, val_ds, test_ds
