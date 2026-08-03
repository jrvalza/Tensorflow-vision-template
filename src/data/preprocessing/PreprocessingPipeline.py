
import tensorflow as tf
from omegaconf import DictConfig


class PreprocessingPipeline:
    """Apply the preprocessing steps configured in the dataset configuration."""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._steps_registry = {
            "pixel-value-normalization": self._normalization_fn
            }

    def _resolve_step(self, step_name: str):
        """
        Resolve a preprocessing step name to its corresponding function.

        Raises:
            ValueError: If the preprocessing step is not registered.
        """
        try:
            return self._steps_registry[step_name]
        except KeyError:
            raise ValueError(f"Unknown preprocessing step: {step_name}")
        
    def _normalization_fn(self, image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Normalize image pixel values by scaling them from [0, 255] to [0, 1]."""

        image = tf.cast(image/255.0, tf.float32)
        return image, label

    def apply(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply the configured preprocessing pipeline to a dataset."""

        process_fns = [
            self._resolve_step(step_name) 
            for step_name in self._cfg.dataset.preprocessing.steps
            ]
        
        if not process_fns:
            return dataset
        
        def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            for fn in process_fns:
                image, label = fn(image, label)
            return image, label

        return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    