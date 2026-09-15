import tensorflow as tf


def pixel_value_normalization(
    image: tf.Tensor, label: tf.Tensor, dtype: str
) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize image pixel values to the [0, 1] range.

    The input image is cast to 'tf.float32' and its pixel values are
    divided by the maximum representable value of the specified data type.
    The label is returned unchanged.

    Args:
        image: Input image tensor.
        label: Label tensor associated with the image.
        dtype: Data type of the input image used to determine its maximum
            representable pixel value.

    Returns:
        A tuple containing the normalized image and the unchanged label.
    """
    image_float = tf.cast(image, tf.float32)
    image_norm = image_float / tf.cast(tf.as_dtype(dtype).max, tf.float32)
    return image_norm, label


def resize_image(
    image: tf.Tensor, label: tf.Tensor, image_size: list[int], task: str
) -> tuple[tf.Tensor, tf.Tensor]:
    """Resize image to the specified size.

    For 'classification' tasks, only the image is resized.
    For 'segmentation' tasks, the mask in 'label' is also resized.

    Args:
        image: Input image tensor.
        label: Class label (classification) or mask tensor (segmentation).
        image_size: Target (height, width).
        task: computer vision task that determines how the label is processed.

    Returns:
        A tuple containing the resized image and label.
    """
    resized_image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR
    )

    if task == "classification":
        return resized_image, label

    elif task == "segmentation":
        resized_mask = tf.image.resize(
            label, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        resized_mask = tf.cast(resized_mask, tf.int32)
        return resized_image, resized_mask
    else:
        raise ValueError(f"Unsupported task: {task!r}")
