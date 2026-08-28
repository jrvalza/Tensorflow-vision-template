import json
from omegaconf import DictConfig

from src.evaluation.evaluators.base_evaluator import BaseEvaluator
from src.evaluation.evaluators.classification_evaluator import ClassificationEvaluator


class EvaluatorManager:
    """Builds a task-specific evaluator based on cfg.task."""

    TASK_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
        "classification": ClassificationEvaluator
    }

    def __init__(self, cfg_evaluation: DictConfig) -> None:
        self._cfg_evaluation = cfg_evaluation

    def __str__(self) -> str:
        """List the available evaluators."""
        evaluators = {
            evaluator_type: evaluator_cls.__name__
            for evaluator_type, evaluator_cls in self.TASK_EVALUATOR_REGISTRY.items()
        }
        return f"Available evaluators:\n{json.dumps(evaluators, indent=4)}"

    def build_evaluator(self) -> BaseEvaluator:
        """Build the evaluator for the task configured in cfg.task.

        Returns:
            An evaluator instance for the configured task.

        Raises:
            ValueError: If cfg.task is not registered.
        """
        try:
            evaluator_cls = self.TASK_EVALUATOR_REGISTRY[self._cfg_evaluation.task]
        except KeyError as e:
            raise ValueError(f"Unknown task: {self._cfg_evaluation.task}") from e
        return evaluator_cls()
