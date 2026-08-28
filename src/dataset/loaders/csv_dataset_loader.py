import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import rasterio
from omegaconf import DictConfig
from collections.abc import Callable

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


class CSVDatasetLoader(BaseDatasetLoader):
    """Loads (image, label) or (image, mask) pairs listed in a single CSV metadata file.

    The CSV has two columns: an image path and a label/mask path, both relative
    to cfg_dataset.root_dir. Rows go to Train/Test by whether the image path
    starts with "Train/" or "Test/"; a fraction of the Train rows is held out
    for validation (cfg_dataset.loader.params.validation_split). Which strategy
    reads the second column is selected by cfg_dataset.task.
    """

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
            names=["image_path", "target"],
        )

        train_rows = df[df["image_path"].str.startswith("Train/")]
        test_rows = df[df["image_path"].str.startswith("Test/")]

        if train_rows.empty or test_rows.empty:
            raise ValueError(
                "The 'metadata_csv' file must contain rows with the prefix 'Train/' and 'Test/'."
            )

        train_rows, val_rows = train_test_split(
            train_rows,
            test_size=self._cfg_dataset.loader.params.validation_split,
            random_state=self._cfg_dataset.loader.params.seed,
        )
        return train_rows, val_rows, test_rows

    def _resolve_classes(self, train_rows: pd.DataFrame) -> None:
        """Resolve class names and indices for the configured task.

        For classification, class names are inferred from the training targets
        and a class-name-to-index lookup table is created. For segmentation,
        class names are obtained from the configuration or generated from
        'num_classes'.

        Args:
            train_rows: DataFrame containing the training targets.
        """
        if self._cfg_dataset.task == "classification":
            self._class_names = sorted(
                train_rows["target"].unique().tolist()
            )  #######################convertir a str?
            self._label_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(self._class_names), tf.range(len(self._class_names))
                ),
                default_value=-1,
            )

        if self._cfg_dataset.task == "segmentation":
            self._class_names = (
                list(self._cfg_dataset.class_names)
                if self._cfg_dataset.class_names is not None
                else [str(i) for i in range(self._cfg_dataset.num_classes)]
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
        index = self._label_table.lookup(target)

        if self._cfg_dataset.loader.params.label_mode == "categorical":
            return tf.one_hot(index, depth=self._num_classes)
        return index

    def _load_segmentation_mask(self, target: tf.Tensor) -> tf.Tensor:
        """Load a segmentation mask raster from the target path.
        Each pixel stores an integer class index.

        Args:
            target: Tensor containing the target path.

        Returns:
            A uint8 tensor containing the per-pixel class indices.
        """
        mask_path = tf.strings.join([self._cfg_dataset.root_dir, "/", target])
        mask = tf.py_function(
            lambda path: _read_raster(path.numpy().decode("utf-8")),
            [mask_path],
            tf.uint8,
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
        full_path = tf.strings.join([self._cfg_dataset.root_dir, "/", image_path])
        image = tf.py_function(
            lambda path: _read_raster(path.numpy().decode("utf-8")),
            [full_path],
            tf.uint8,
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
            (df["image_path"].tolist(), df["target"].tolist())
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
