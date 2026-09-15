"""Central registry of evaluation tasks.

To add a new evaluation task, implement it in 'src/evaluation' and
register it here under a unique string key. That key is what users
reference from YAML config, and is validated against this registry by
the corresponding configuration schema.
"""

from src.evaluation.evaluators.base_evaluator import BaseEvaluator
from src.evaluation.evaluators.classification_evaluator import ClassificationEvaluator

EVALUATOR_TASKS_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "classification": ClassificationEvaluator
}
