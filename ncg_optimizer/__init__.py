"""ncg-optimizer -- a set of optimizer about nonliear conjugate gradient in Pytorch.
API and usage patterns are the same as `torch.optim`__
Example
-------
>>> import torch_optimizer as optim
# model = ...
>>> optimizer = optim.PRP(model.parameters())
>>> optimizer.step()
"""
from typing import Dict, List, Type

from torch.optim.optimizer import Optimizer

from .lcg import LCG

__all__ = (
    'LCG',
)
__version__ = '0.0.1b0'


_package_opts = [
    LCG
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