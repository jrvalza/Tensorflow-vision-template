
import hydra
from omegaconf import DictConfig, OmegaConf
from keras.layers import Conv2D
from keras.layers import Input

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from src.utils.utils import Config
from src.data_loaders.DatasetManager import DatasetManager

# Definición de la arquitectura según el modo funcional
def deep_CNN(width, height, depth, classes, batchNorm):

    # Definimos entradas en modo "channels last"
    inputs = Input(shape=(height, width, depth)) #(X)

    # Definimos la arquitectura
    # Primer set de capas
    x1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs) #(X)
    if batchNorm:
        x1 = BatchNormalization()(x1) #(X)

    x1 = Conv2D(32, (3, 3), padding="same", activation="relu")(x1) #(X)
    if batchNorm:
        x1 = BatchNormalization()(x1) #(X)

    x1 = MaxPooling2D(pool_size=(2, 2))(x1) #(X)
    x1 = Dropout(0.25)(x1) #(X)

    # Segundo set de capas
    x2 = Conv2D(64, (3, 3), padding="same", activation="relu")(x1) #(X)
    if batchNorm: #(X)
        x2 = BatchNormalization()(x2) #(X)

    x2 = Conv2D(64, (3, 3), padding="same", activation="relu")(x2) #(X)
    if batchNorm:
        x2 = BatchNormalization()(x2) #(X)

    x2 = MaxPooling2D(pool_size=(2, 2))(x2) #(X)
    x2 = Dropout(0.25)(x2) #(X)

    # Top model
    xfc = Flatten()(x2) #(X)

    xfc = Dense(units=512, activation='relu')(xfc)
    if batchNorm:
        xfc = BatchNormalization()(xfc)
    xfc = Dropout(0.5)(xfc) #(X)

    # Clasificador softmax
    predictions = Dense(units=classes, activation='softmax')(xfc)

    # Unimos las entradas y el modelo mediante la función Model con parámetros inputs y ouputs (Consultar la documentación)
    model = Model(inputs=inputs, outputs=predictions) #(X)

    # La función debe devolver el modelo como salida
    return model



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(schema, cfg)
    print(OmegaConf.to_yaml(cfg))

    #LOAD DATA
    manager = DatasetManager(cfg)

    train_ds, val_ds, test_ds = manager.load_data()
    labelNames = manager.class_names

    #TRAIN
    # Compilar el modelo
    print("[INFO]: Compilando el modelo...")
    # Instanciamos el modelo invocando la función que hemos creado
    model = deep_CNN(width=32, height=32, depth=4, classes=len(labelNames), batchNorm=True)

    # Compilamos el modelo. Ver método Model.compile en la documentación. Emplear el optimizador SGD con un learning rate de 0.01. Entender el método de pérdidas "loss".
    my_opt = SGD(learning_rate=0.01)
    model.compile(optimizer=my_opt, loss="categorical_crossentropy", metrics=["accuracy"])


    # Entrenamiento de la red con 5 épocas. Ver documentación del método Model.fit.
    print("[INFO]: Entrenando la red...")
    H = model.fit(train_ds,
                validation_data=val_ds,
                epochs=2)

if __name__ == "__main__":
    main()
