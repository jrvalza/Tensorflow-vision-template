import tensorflow as tf
from tensorflow.keras.models import Model

from src.evaluation.metrics import Metrics
from src.evaluation.plotter import Plotter
from src.config.registry import EvaluatorRegistries


class EvaluatorManager:
    """Selects and invokes the evaluator configured for a specific task."""

    def __init__(self, task: str, evaluator_registries: EvaluatorRegistries) -> None:
        """Initialize the evaluator manager.

        Args:
            task: Computer vision task used to select the corresponding evaluator
                from the registry.
            evaluator_registries: Registry containing the evaluators available for
                each supported task.
        """
        self._evaluator_cls = evaluator_registries.task_evaluators[task]

    def evaluate(
        self,
        model: Model,
        test_ds: tf.data.Dataset,
        class_names: list[str],
        label_mode: str,
    ) -> None:
        """Evaluate a trained model using the task-specific evaluator.

        Args:
            model: Trained Keras model to evaluate.
            test_ds: Batched test dataset used for evaluation.
            class_names: Class names, in the order used by the model's output.
            label_mode: Label encoding used by the dataset, e.g. 'categorical', 'binary', or 'int'.
        """
        evaluator = self._evaluator_cls(
            model=model,
            test_ds=test_ds,
            class_names=class_names,
            label_mode=label_mode,
            metrics=Metrics(),
            plotter=Plotter(),
        )
        evaluator.evaluate()
