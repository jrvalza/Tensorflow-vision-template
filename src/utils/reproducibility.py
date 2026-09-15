import tensorflow as tf


def set_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and TensorFlow RNGs, and force deterministic GPU ops"""
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
