import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.Trainer import Trainer
from src.model.ModelManager import ModelManager
from src.data.DatasetManager import DatasetManager
from src.evaluation.EvaluatorManager import EvaluatorManager

from src.utils.dataclasses import TrainConfig
from src.evaluation.plots import plot_training_curves


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):

    schema = OmegaConf.structured(TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    print(OmegaConf.to_yaml(cfg))

    # LOAD DATA
    print("[INFO]: Cargando datos...")
    data_manager = DatasetManager(cfg)
    train_ds, val_ds, test_ds = data_manager.load_data()

    # TRAIN
    print("[INFO]: Compilando el modelo...")
    model_manager = ModelManager(cfg)
    image_size = tuple(cfg.dataset.loader.params.image_size)
    input_shape = (*image_size, cfg.dataset.num_bands)
    model = model_manager.build(
        input_shape=input_shape, num_classes=data_manager.num_classes
    )

    print("[INFO]: Entrenando el modelo...")
    trainer = Trainer(cfg, model, train_ds, val_ds)
    trainer.train()
    plot_training_curves(trainer._history)

    print("[INFO]: Evaluando el modelo...")
    evaluator_manager = EvaluatorManager(cfg)
    evaluator = evaluator_manager.build_evaluator()
    evaluator.evaluate(trainer._model, test_ds, class_names=data_manager.class_names)

    print("[INFO]: Fin.")


if __name__ == "__main__":
    main()
