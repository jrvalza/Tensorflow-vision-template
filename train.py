
import hydra
from omegaconf import DictConfig, OmegaConf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model

from src.utils.dataclasses import Config
from src.model.ModelManager import ModelManager
from src.data.DatasetManager import DatasetManager


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(schema, cfg)
    print(OmegaConf.to_yaml(cfg))

    #LOAD DATA
    data_manager = DatasetManager(cfg)

    train_ds, val_ds, test_ds = data_manager.load_data()
    labelNames = data_manager.class_names

    #TRAIN
    # Compilar el modelo
    print("[INFO]: Compilando el modelo...")

    input_shape = (32, 32, 4)
    
    model_manager = ModelManager(cfg)
    model = model_manager.build(input_shape=input_shape,num_classes=data_manager.num_classes)

    my_opt = SGD(learning_rate=0.01)
    model.compile(optimizer=my_opt, loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    print()
    print(f"Model parameters: {model.count_params():,}")
    plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
    )

    """
    # Entrenamiento de la red con 5 épocas. Ver documentación del método Model.fit.
    print("[INFO]: Entrenando la red...")
    H = model.fit(train_ds,
                validation_data=val_ds,
                epochs=2)
    """

if __name__ == "__main__":
    main()
