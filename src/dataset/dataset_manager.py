import tensorflow as tf

from src.utils.paths import get_cache_dir
from src.config.registry import DatasetRegistries
from src.config.schemes.dataset_scheme import DatasetConfig
from .preprocessing.augmentation_pipeline import AugmentationPipeline
from .preprocessing.preprocessing_pipeline import PreprocessingPipeline


class DatasetManager:
    """Loads, preprocesses, augments, and caches the configured datasets.

    The dataset loader, preprocessing pipeline, and augmentation pipeline
    are selected from the provided registries and configured through the
    dataset configuration.
    """

    def __init__(
        self,
        cfg_dataset: DatasetConfig,
        dataset_registries: DatasetRegistries,
    ) -> None:
        """Initialize the dataset manager.

        Args:
            cfg_dataset: Dataset configuration containing the loader,
                preprocessing, augmentation, and related settings.
            dataset_registries: Registries of available dataset loaders,
                preprocessing steps, and augmentation steps, injected as
                a single dependency.
        """

        self._cfg_dataset = cfg_dataset
        self._loader = dataset_registries.loaders[self._cfg_dataset.loader.name](
            self._cfg_dataset
        )
        self._preprocess_pipeline = PreprocessingPipeline(
            self._cfg_dataset, dataset_registries.preprocessing_steps
        )
        self._augmentation_pipeline = AugmentationPipeline(
            self._cfg_dataset, dataset_registries.augmentation_steps
        )

    def __repr__(self) -> str:
        return (
            f"DatasetManager(loader={self._cfg_dataset.loader.name}, "
            f"preprocessing={self._cfg_dataset.preprocessing.enabled}, "
            f"augmentation={self._cfg_dataset.augmentation.enabled})"
        )

    @property
    def num_classes(self) -> int | None:
        """Return the number of classes in the dataset, if available."""
        return self._loader.num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Return the dataset class names, if available."""
        return self._loader.class_names

    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare the train, validation, and test datasets.

        The datasets are loaded using the configured loader, optionally
        preprocessed and augmented according to the configuration, cached
        on disk, and prefetched using 'tf.data.AUTOTUNE'. Data
        augmentation is applied only to the training dataset.

        Returns:
            A tuple containing the training, validation, and test datasets.
        """
        train_ds, val_ds, test_ds = self._loader.load_data()

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------
        if self._cfg_dataset.preprocessing.enabled:
            train_ds = self._preprocess_pipeline.apply(train_ds)
            val_ds = self._preprocess_pipeline.apply(val_ds)
            test_ds = self._preprocess_pipeline.apply(test_ds)

        # --------------------------------------------------
        # Cache preprocessed datasets
        # --------------------------------------------------
        cache_dir = get_cache_dir()
        train_ds = train_ds.cache(str(cache_dir / "train"))
        val_ds = val_ds.cache(str(cache_dir / "val"))
        test_ds = test_ds.cache(str(cache_dir / "test"))

        # --------------------------------------------------
        # Data augmentation
        # --------------------------------------------------
        if self._cfg_dataset.augmentation.enabled:
            train_ds = self._augmentation_pipeline.apply(train_ds)

        # --------------------------------------------------
        # Prefetch
        # --------------------------------------------------
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds
