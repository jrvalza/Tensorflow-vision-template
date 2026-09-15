from dataclasses import dataclass
from collections.abc import Callable

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Optimizer

from src.dataset.preprocessing.base_pipeline import Step
from src.evaluation.evaluators.base_evaluator import BaseEvaluator
from src.dataset.loaders.base_dataset_loader import BaseDatasetLoader


from .registries.dataset_registries import (
    LOADER_REGISTRY,
    PREPROCESSING_STEPS_REGISTRY,
    AUGMENTATION_STEPS_REGISTRY,
)

from .registries.model_registries import (
    MODEL_BUILDERS_REGISTRY,
    MODEL_BLOCKS_REGISTRY,
    MODEL_POOLING_LAYERS_REGISTRY,
)

from .registries.training_registries import (
    TRAINING_OPTIMIZERS_REGISTRY,
    TRAINING_LOSSES_REGISTRY,
    TRAINING_CALLBACKS_REGISTRY,
)

from .registries.evaluator_registries import (
    EVALUATOR_TASKS_REGISTRY,
)


# ================================================DATASET================================================
@dataclass
class DatasetRegistries:
    """Bundles the registries DatasetManager needs, injected as one unit."""

    loaders: dict[str, type[BaseDatasetLoader]]
    preprocessing_steps: dict[str, Callable[..., Step]]
    augmentation_steps: dict[str, Callable[..., Layer]]


DATASET_REGISTRIES = DatasetRegistries(
    loaders=LOADER_REGISTRY,
    preprocessing_steps=PREPROCESSING_STEPS_REGISTRY,
    augmentation_steps=AUGMENTATION_STEPS_REGISTRY,
)


# =================================================MODEL=================================================
@dataclass
class ModelRegistries:
    """Bundles the registries ModelManager needs, injected as one unit."""

    builders: dict[str, Callable[..., Model]]
    blocks: dict[str, Callable[..., tf.Tensor]]
    pooling_layers: dict[str, Callable[..., Layer]]


MODEL_REGISTRIES = ModelRegistries(
    builders=MODEL_BUILDERS_REGISTRY,
    blocks=MODEL_BLOCKS_REGISTRY,
    pooling_layers=MODEL_POOLING_LAYERS_REGISTRY,
)


# ===============================================TRAINING================================================
@dataclass
class TrainingRegistries:
    """Bundles the registries TrainingManager needs, injected as one unit."""

    optimizers: dict[str, type[Optimizer]]
    losses: dict[str, type[Loss]]
    callbacks: dict[str, type[Callback]]


TRAINING_REGISTRIES = TrainingRegistries(
    optimizers=TRAINING_OPTIMIZERS_REGISTRY,
    losses=TRAINING_LOSSES_REGISTRY,
    callbacks=TRAINING_CALLBACKS_REGISTRY,
)


# ==============================================EVALUATION===============================================
@dataclass
class EvaluatorRegistries:
    """Bundles the registries EvaluatorManager needs, injected as one unit."""

    task_evaluators: dict[str, type[BaseEvaluator]]


EVALUATOR_REGISTRIES = EvaluatorRegistries(task_evaluators=EVALUATOR_TASKS_REGISTRY)


# ===============================================GROUPING================================================
@dataclass
class ConfigRegistries:
    """Groups the registries used to validate configurable components."""

    dataset: DatasetRegistries
    model: ModelRegistries
    training: TrainingRegistries
    evaluator: EvaluatorRegistries
