"""Entry point for the end-to-end model training pipeline.

Composes the Hydra config, validates it with Pydantic against the
registered components, and runs load → build → train → evaluate.
"""

import json
import hydra
from omegaconf import DictConfig, OmegaConf

from src.config.registry import (
    ConfigRegistries,
    DATASET_REGISTRIES,
    MODEL_REGISTRIES,
    TRAINING_REGISTRIES,
    EVALUATOR_REGISTRIES,
)
from src.config.scheme import TrainConfig
from src.config.params_resolver import params_resolvers

from src.utils.paths import get_checkpoint_dir
from src.utils.reproducibility import set_reproducibility

from src.model.model_manager import ModelManager
from src.dataset.dataset_manager import DatasetManager
from src.training.training_manager import TrainingManager
from src.evaluation.evaluator_manager import EvaluatorManager


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """Run the end-to-end training pipeline: validate config, load data,
    build the model, train it and evaluate it on the test split.

    Args:
        cfg: Hydra-composed configuration, validated against
            src.config.scheme.TrainConfig before use.
    """

    params_resolvers()
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

    # CONFIGURATION VALIDATION WITH PYDANTIC
    config_registries = ConfigRegistries(
        dataset=DATASET_REGISTRIES,
        model=MODEL_REGISTRIES,
        training=TRAINING_REGISTRIES,
        evaluator=EVALUATOR_REGISTRIES,
    )

    cfg_validated = TrainConfig.model_validate(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        context={"registries": config_registries},
    )

    # FIXING REPRODUCIBILITY
    set_reproducibility(cfg_validated.global_seed, cfg_validated.reproducibility)

    # LOAD DATA
    print("[INFO]: Cargando datos...")
    data_manager = DatasetManager(cfg_validated.dataset, DATASET_REGISTRIES)

    train_ds, val_ds, test_ds = data_manager.load_data()

    class_names_path = str(get_checkpoint_dir() / "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(data_manager.class_names, f, indent=2)

    # TRAIN
    print("[INFO]: Creando el modelo...")
    model_manager = ModelManager(cfg_validated.model, MODEL_REGISTRIES)
    image_size = tuple(cfg_validated.dataset.loader.params.image_size)
    input_shape = (*image_size, cfg_validated.dataset.num_bands)
    model = model_manager.build(
        input_shape=input_shape, num_classes=data_manager.num_classes
    )

    print("[INFO]: Entrenando el modelo...")
    trainer = TrainingManager(
        cfg_validated.training, model, train_ds, val_ds, TRAINING_REGISTRIES
    )
    model = trainer.train()

    print("[INFO]: Evaluando el modelo...")
    evaluator_manager = EvaluatorManager(cfg_validated.task, EVALUATOR_REGISTRIES)
    evaluator_manager.evaluate(
        model,
        test_ds,
        class_names=data_manager.class_names,
        label_mode=cfg_validated.dataset.loader.params.label_mode,
    )

    print("[INFO]: Fin.")


if __name__ == "__main__":
    main()
