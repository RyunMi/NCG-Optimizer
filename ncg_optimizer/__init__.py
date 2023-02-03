"""ncg-optimizer -- a set of optimizer about nonliear conjugate gradient in Pytorch.
API and usage patterns are the same as `torch.optim`__
Example
-------
>>> import ncg_optimizer as optim
>>> optimizer = optim.LCG(model.parameters(), eps=1e-5)
>>> def closure():
>>>     optimizer.zero_grad()
>>>     loss_fn(model(input), target).backward()
>>>     return loss_fn
>>> optimizer.step(closure)
"""
from typing import Dict, List, Type

from torch.optim.optimizer import Optimizer

from .lcg import LCG
from .fr import FR

__all__ = (
    'LCG',
    'FR',
)
__version__ = '0.0.2b'


_package_opts = [
    LCG,
    FR
]  # type: List[Type[Optimizer]]


_NAME_OPTIM_MAP = {
    opt.__name__.lower(): opt for opt in _package_opts
}  # type: Dict[str, Type[Optimizer]]


def get(name: str) -> Type[Optimizer]:
    r"""Returns an optimizer class from its name. Case insensitive.
    Args:
        name: the optimizer name.
    """
    optimizer_class = _NAME_OPTIM_MAP.get(name.lower())
    if optimizer_class is None:
        raise ValueError('Optimizer {} not found'.format(name))
    return 