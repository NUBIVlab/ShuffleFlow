_PRIOR = {}

def register_prior(cls=None, *, name=None):
    """A decorator for registering prior classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PRIOR:
            raise ValueError(f'Already registered prior with name: {local_name}')
        _PRIOR[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_prior(name: str, **kwargs):
    if _PRIOR.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return _PRIOR[name](**kwargs)
 
from .l2_score import *
from .sds import *
from .surrogate_score import *
from .tv import *
