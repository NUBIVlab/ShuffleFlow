import hydra
from hydra.utils import instantiate

from data_generators import *


@hydra.main(version_base="1.3", config_path="configs", config_name="generate_data")
def generate_data(config):
    data_generator = instantiate(config.inverse_problem.data_generator, _recursive_=False)
    data_generator.generate_data()
    
    
if __name__ == "__main__":
    generate_data()