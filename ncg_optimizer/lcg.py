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
        >>> optimizer = optim.LCG(model.parameters(), eps=1e-5)
        >>> def closure():
        >>>     optimizer.zero_grad()
        >>>     loss_fn(model(input), target).backward()
        >>>     return loss_fn
        >>> optimizer.step(closure)
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

                n=len(d_p)

                state = self.state[p]

                if len(state) == 0:
                    # Grade of quadratic functions
                    # i.e. The allowance of the linear equation
                    state['r'] = copy.deepcopy(d_p.data)

                    if torch.norm(state['r']) < group['eps']:
                        # Stop condition
                        return loss
                    
                    # Direction vector
                    state['pb'] = copy.deepcopy(-d_p.data)
                    
                    # Coefficient matrix
                    state['A'] = torch.stack(
                        [torch.autograd.grad(
                            d_p[i],
                            p, 
                            grad_outputs=torch.ones_like(d_p[i]),
                            retain_graph=True)[0]
                        for i in range(0,n)])
                else:
                    state['r'] = copy.deepcopy(d_p.data)
                    
                    if torch.norm(state['r']) < group['eps']:
                        return loss
                    
                    # Parameters that make gradient steps
                    beta = torch.dot(state['r'], state['r']) / state['rdotr']

                    state['pb'] = -state['r'] + beta * state['pb']

                r, pb = state['r'], state['pb']

                state['rdotr'] = torch.dot(r, r)
                
                z = torch.matmul(state['A'], pb)

                # Step factor
                alpha = state['rdotr'] / torch.matmul(pb, z)

                p.data.add_(pb, alpha=alpha)

        return loss