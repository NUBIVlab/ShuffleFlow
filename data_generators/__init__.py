import logging
from abc import ABC, abstractmethod

from omegaconf import DictConfig

from utils.logger import setup_logger
from utils.name import *

_DATA_GENERATORS = {}


def register_generator(cls=None, *, name=None):
  """A decorator for registering data generator classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _DATA_GENERATORS:
      raise ValueError(f'Already registered data generator with name: {local_name}')
    _DATA_GENERATORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_generator(name: str, **kwargs):
    if _DATA_GENERATORS.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return _DATA_GENERATORS[name](**kwargs)


class DataGenerator(ABC):
    def __init__(self, inverse_problem_config, data_config):
      # self.config = config
      self.name_data = get_name_data(data_config)
      self.name_operator = get_name_operator(inverse_problem_config)
      self.data_path = data_config.data_path
      self.inverse_problem_config = inverse_problem_config
      self.data_config = data_config
      # Set up logger.
      self.logger = setup_logger("Data Generator", level=logging.INFO)

    @abstractmethod
    def generate_data(self):
        pass
    

class BaseDataGenerator(ABC):
    def __init__(self, operator: DictConfig, dataset: DictConfig, n_vis: int=20, **kwargs):
        # self.inverse_problem_config = inverse_problem_config
        self.operator_config = operator
        self.data_config = dataset
        self.name_data = get_name_data(self.data_config)
        self.name_operator = get_name_operator(self.operator_config)
        self.data_path = self.data_config.data_path
        self.logger = setup_logger("Data Generator", level=logging.INFO)
        self.n_vis = n_vis

    @abstractmethod
    def generate_data(self):
        pass
      
    
from .coded_diffraction_pattern import CodedDiffractionPatternDataGenerator
from .deblur import DeblurDataGenerator
from .mri import MRIDataGenerator
from .phase_retrieval import PhaseRetrievalDataGenerator
