
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.utils import Config

from src.data_loaders.DatasetManager import DatasetManager


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(schema, cfg)
    print(OmegaConf.to_yaml(cfg))

    #LOAD DATA
    manager = DatasetManager(cfg)
    manager.load_data()
    print(manager.num_classes)
    print(manager.class_names)

    #PREPROCESSING DATA
    
if __name__ == "__main__":
    main()
