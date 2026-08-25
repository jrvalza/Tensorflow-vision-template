import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow RNGs, and force deterministic GPU ops"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
