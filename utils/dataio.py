import json
import logging
import os

import h5py
import hdf5storage
import numpy as np
import torch
import yaml

from utils.name import *


def load_galaxy_file(file:str) -> dict:
    """Load galaxy data from `.hdf5` file.

    Args:
        file (str): Path to file.

    Returns:
        dict: Galaxy data.
    """
    logger = logging.getLogger('DataIO')
    data = h5py.File(file, 'r')
    logger.debug(' Successfully loaded data from "%s".', file)
    return data


def load_fastMRI_file(data_path:str, file_idx:int) -> dict:
    """Load fastMRI data from `.h5` file.

    Args:
        file (str): Path to file.
        file_idx (int): Index of the file.

    Returns:
        dict: FastMRI data.
    """
    logger = logging.getLogger('DataIO')
    file = os.path.join(data_path, f'file{file_idx}.h5')
    data = h5py.File(file, 'r')
    logger.debug(' Successfully loaded data from "%s".', file)
    return data



def load_config(file:str) -> dict:
    """Load config files in `.yaml` format.

    Args:
        file (str): Path to file.

    Returns:
        dict: Dictionary of data.
    """
    logger = logging.getLogger('DataIO')
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    logger.debug(' Successfully loaded data from "%s".', file)
    return config
    

def load_mat(file:str) -> np.ndarray:
    """Load data in `.mat` files.

    Args:
        file (str): Path to file.

    Returns:
        tuple: Tuple of data cubes converted to numpy arrays.
    """
    logger = logging.getLogger('DataIO')
    
    dict = h5py.File(file)
    keys = list(dict.keys())

    if len(keys) == 1:
        data = np.array(dict[keys[0]])
    else:
        data = ()
        for key in keys:
            data += (np.array(dict[key]), )
    logger.debug(' Successfully loaded data from "%s".', file)
    return data
    

def save_mat(file:str, data:np.ndarray, key:str='data') -> None:
    """Save data to `.mat` file.

    Args:
        file (str): Path to file.
        data (np.ndarray): The dictionary of data to be saved.
        key (str, optional): The key to be used in the dictionary. Defaults to `'data'`.
    """
    logger = logging.getLogger('DataIO')
    if os.path.exists(file):
        os.remove(file)
    hdf5storage.savemat(file, {key: data})
    logger.debug(' Successfully saved data to "%s".', file)


def save_log(results_path:str, log:dict) -> None:
    """Save log dictionary to `.json` file.

    Args:
        results_path (str): Path to results directory.
        log (dict): Log dictionary.
    """
    logger = logging.getLogger('DataIO')
    log_file = os.path.join(results_path, 'log.json')
    with open(log_file, 'w') as f:
        json.dump(log, f)
    logger.debug(' Successfully saved log to "%s".', log_file)
    
    
def load_log(log_file:str) -> dict:
    """Load log dictionary from `.json` file.

    Args:
        log_file (str): Path to log file.

    Returns:
        dict: Log dictionary.
    """
    logger = logging.getLogger('DataIO')
    with open(log_file, 'r') as f:
        log = json.load(f)
    logger.debug(' Successfully loaded log file from "%s".', log_file)
    return log


def load_results_vi(config: DictConfig, data_idx: int) -> tuple:
    name_data, name_operator, name_vi = get_name_data(config.inverse_problem.dataset), get_name_operator(config.inverse_problem.operator), get_name_vi(config)
    for n_epochs in list(range(config.n_epochs, 0, -1000)):
        try:
            result_path = os.path.join(config.result_path, config.inverse_problem.name, name_data, name_operator, config.flow.name, name_vi, f'{data_idx}', f'{n_epochs}epochs')
            results = torch.load(os.path.join(result_path, f'results_{config.num_samples}samples.pth'), weights_only=False)
            log = load_log(os.path.join(result_path, 'log.json'))
            return results, log
        except:
            pass
    assert False, f'No results found for {result_path}.'
    
    
def load_results_ps(config: DictConfig, data_idx: int) -> tuple:
    name_data, name_operator, name_sampler = get_name_data(config.inverse_problem.dataset), get_name_operator(config.inverse_problem.operator), get_name_sampler(config.sampler)
    result_path = os.path.join(config.result_path, config.inverse_problem.name, name_data, name_operator, config.sampler.name, name_sampler, f'{data_idx}')
    results = torch.load(os.path.join(result_path, f'results_{config.num_samples}samples.pth'), weights_only=False)
    log = load_log(os.path.join(result_path, 'log.json'))
    return results, log
