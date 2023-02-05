"""ncg-optimizer -- a set of optimizer about nonliear conjugate gradient in Pytorch.
API and usage patterns are the same as `torch.optim`__
Example
-------
>>> import ncg_optimizer as optim
>>> optimizer = optim.PRP(
>>>     model.parameters(), eps = 1e-3, 
>>>     line_search = 'Armijo', c1 = 1e-4, c2 = 0.4,
>>>     lr = 1, rho = 0.5, eta = 5,  max_ls = 10)
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
from .prp import PRP
from .hs import HS
from .cd import CD
from .dy import DY
from .ls import LS
from .hz import HZ
from .hs_dy import HS_DY

__all__ = (
    'LCG',
    'FR',
    'PRP',
    'HS',
    'CD',
    'DY',
    'LS',
    'HZ',
    'HS_DY'
)
__version__ = '0.0.4'


_package_opts = [
    LCG,
    FR,
    PRP,
    HS,
    CD,
    DY,
    LS,
    HZ,
    HS_DY,
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