import torch
from torch.optim.optimizer import Optimizer

from typing import List, Optional

__all__ = ('LCM',)

class LCM(Optimizer):
    def __init__(
        self,
        params,
        eps=1e-10,
    ):
        if eps < 0.0:
                raise ValueError('Invalid epsilon value: {}'.format(eps))
        defaults = dict(
            eps=eps,
        )
        super(LCM, self).__init__(params, defaults)