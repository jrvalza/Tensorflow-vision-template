"""Shared literals for configuration schemas."""

from typing import Literal

literal_verbose_fit = Literal[0, 1, 2]

literal_image_dtypes = Literal["uint8"]

literal_metrics = list[Literal["accuracy"]]

literal_label_modes = Literal["int", "categorical"]

literal_cv_tasks = Literal["classification", "segmentation"]
