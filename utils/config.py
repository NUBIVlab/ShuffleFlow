from omegaconf.dictconfig import DictConfig


def set_config(config: DictConfig, value_name: str, value: float) -> DictConfig:
    if value_name in config.keys():
        config[value_name] = value
        return config
    else:
        for key in config.keys():
            try:
                config[key] = set_config(config[key], value_name, value)
                return config
            except:
                pass
    assert False, f'Value name {value_name} not found in config.'


def set_config_absolute(config: DictConfig, value_name: str, value: float) -> DictConfig:
    if isinstance(value_name, list):
        if len(value_name) > 1:
            config[value_name[0]] = set_config_absolute(config[value_name[0]], value_name[1:], value)
        else:
            config[value_name[0]] = value
    else:
        config[value_name] = value
    return config