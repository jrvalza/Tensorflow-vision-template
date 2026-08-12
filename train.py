import json
import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.Trainer import Trainer
from src.model.BuilderModelManager import BuilderModelManager
from src.data.DatasetManager import DatasetManager
from src.evaluation.EvaluatorManager import EvaluatorManager

from src.utils.dataclasses import TrainConfig
from src.utils.paths import get_checkpoint_dir
from src.evaluation.plots import plot_training_curves


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):

    schema = OmegaConf.structured(TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    print(OmegaConf.to_yaml(cfg))

    # LOAD DATA
    print("[INFO]: Cargando datos...")
    data_manager = DatasetManager(cfg.dataset)
    train_ds, val_ds, test_ds = data_manager.load_data()

    class_names_path = str(get_checkpoint_dir() / "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(data_manager.class_names, f, indent=2)

    # TRAIN
    print("[INFO]: Creando el modelo...")
    model_manager = BuilderModelManager(cfg.model)
    image_size = tuple(cfg.dataset.loader.params.image_size)
    input_shape = (*image_size, cfg.dataset.num_bands)
    model = model_manager.build(
        input_shape=input_shape, num_classes=data_manager.num_classes
    )

    print("[INFO]: Entrenando el modelo...")
    trainer = Trainer(cfg.training, model, train_ds, val_ds)
    model = trainer.train()
    plot_training_curves(trainer.history)

    print("[INFO]: Evaluando el modelo...")
    evaluator_manager = EvaluatorManager(cfg)
    evaluator = evaluator_manager.build_evaluator()
    evaluator.evaluate(model, test_ds, class_names=data_manager.class_names)

    print("[INFO]: Fin.")


if __name__ == "__main__":
    main()
