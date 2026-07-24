
import tensorflow as tf
from omegaconf import DictConfig

class PreprocessingPipeline:

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._steps_registry = {
            "pixel-value-normalization": self._normalization_fn
            }

    def _resolve_step(self, step_name):
        try:
            return self._steps_registry[step_name]
        except KeyError:
            raise ValueError(f"Unknown preprocessing step: {step_name}")
        
    def _normalization_fn(self, image, label):
        image = tf.cast(image/255., tf.float32)
        return image, label
    
    def apply(self, dataset):

        process_fns = [
            self._resolve_step(step_name) 
            for step_name in self._cfg.dataset.preprocessing.steps
            ]
        
        if not process_fns:
            return dataset
        
        def preprocess(image, label):
            for fn in process_fns:
                image, label = fn(image, label)
            return image, label

        return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    