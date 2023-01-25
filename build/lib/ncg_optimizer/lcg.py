import torch
from torch.optim.optimizer import Optimizer

import copy

__all__ = ('LCG',)

class LCG(Optimizer):
    r"""Implements Linear Conjugate Gradient.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        eps: term added to the denominator to improve
            numerical stability (default: 1e-5)
    
    Example:
        >>> import ncg_optimizer as optim
        >>> optimizer = optim.LCG(model.parameters(), eps=1e-8)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        eps=1e-5,
    ):
        if eps < 0.0:
                raise ValueError('Invalid epsilon value: {}'.format(eps))
        defaults = dict(
            eps=eps,
        )

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

                state = self.state[p]

                if len(state) == 0:
                    # Grade of quadratic functions
                    # i.e. The allowance of the linear equation
                    state['r'] = copy.deepcopy(d_p.data)
                    # Negative grade of quadratic functions
                    state['pb'] = copy.deepcopy(-d_p.data)
                    # Coefficient matrix
                    state['A'] = torch.autograd.grad(d_p, p)
                
                if state['r'] < group['eps']:
                    # Stop condition
                    break
                
                r, pb = state['r'], state['pb']

                rdotr = torch.dot(r, r)
                z = torch.dot(state['A'], pb)
                
                # Step factor
                alpha = rdotr / torch.dot(pb, z)

                p = p.data.add_(pb, alpha=alpha)

                state['r'] = p.grad.data

                # Parameters that make gradient steps
                beta = torch.dot(state['r'], state['r']) / rdotr

                state['pb'] = -state['r'].add_(-pb, beta) 

        return loss