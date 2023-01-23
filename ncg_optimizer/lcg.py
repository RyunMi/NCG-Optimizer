import torch
from torch.optim.optimizer import Optimizer

from typing import List, Optional

__all__ = ('LCG',)

class LCG(Optimizer):
    r"""Implements Linear Conjugate Gradient.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
    
    Example:
        >>> import ncg_optimizer as optim
        >>> optimizer = optim.LCG(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

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

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))
        super(LCG, self).__init__(params, defaults)


    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
        
        return loss


def lcg(params,grad):
    return 
   
def _linear_equation(func,params):
    f_grads = torch.autograd.grad(func, params, create_graph=True)
    return f_grads

def _get_A(f_grad,params):
    f_grads2 = torch.autograd.grad(f_grad, params)
    return f_grads2

