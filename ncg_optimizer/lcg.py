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
            numerical stability (default: 1e-8)
    
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
        eps=1e-8,
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
                
                grad = p.grad.data

                # Coefficient matrix
                A = p.grad.grad.data

                state = self.state[p]

                if len(state) == 0:
                    # Grade of quadratic functions
                    # i.e. The allowance of the linear equation
                    state['r'] = copy.deepcopy(grad)
                    # Negative grade of quadratic functions
                    state['pb'] = copy.deepcopy(-grad)
                    # Parameters that make gradient steps
                    state['beta'] == 0
                
                rdotr = torch.dot(state['r'], state['r'])
                z = torch.dot(A, state['pb'])
                
                # Step factor
                alpha = rdotr / torch.dot(state['pb'], z)

                p = p.add_(-p.grad, alpha)

                state['r'] = p.grad.data

                if state['r'] < group['eps']:
                    # Stop condition
                    break

                state['beta'] = torch.dot(state['r'], state['r']) / rdotr

                state['pb'] = -state['r'].add_(-state['pb'], state['beta']) 

        return loss