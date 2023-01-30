import torch

from torch.optim.optimizer import Optimizer

from Line_Search import Strong_Wolfe
from Line_Search import General_Wolfe
from Line_Search import General_Armijo

import copy

import warnings

__all__ = ('FR',)

class FR(Optimizer):
    r"""Implements Fletcher-Reeves Conjugate Gradient.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        eps: term added to the denominator to improve
            numerical stability (default: 1e-3)
        line_search: designates line search to use (default: 'Strong_Wolfe')
            Options:
                'None': uses exact line search(requires the loss is quadratic)
                'Strong_Wolfe': uses Strong_Wolfe bracketing line search
                'General_Wolfe': uses General_Wolfe bracketing line search
                'General_Armijo': uses General_Armijo bracketing line search
        c1: sufficient decrease constant in (0, 1) (default: 1e-4)
        c2: curvature condition constant in (0, 1) (default: 0.1)
        lr: initial step length of Line Search (default: 1)
        rho: contraction factor of Line Search (default: 0.5)
        eta: secondary adjustment factor of Wolfe Line Search (default: 0.5)
        'max_ls': maximum number of line search steps permitted (default: 10)
    
    Example:
        >>> import ncg_optimizer as optim
        >>> optimizer = optim.FR(
        >>>     model.parameters(), eps = 1e-3, 
        >>>     line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.1,
        >>>     sigma = 1, rho = 0.5, eta = 0.5)
        >>> def closure():
        >>>     optimizer.zero_grad()
        >>>     loss_fn(model(input), target).backward()
        >>>     return loss_fn
        >>> optimizer.step(closure)
    """

    def __init__(
        self,
        params,
        eps = 1e-3,
        line_search = 'Strong_Wolfe',
        c1 = 1e-4,
        c2 = 0.1,
        lr = 1,
        rho = 0.5,
        eta = 0.5,
        max_ls = 10
    ):
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))

        if line_search not in [
            'Strong_Wolfe', 
            'General_Wolfe',
            'General_Armijo',
            'None',
            ]:
            raise ValueError("Invalid line search: {}".format(line_search))
        elif line_search == 'None':
            warnings.warn("Unless loss is a quadratic function, this is not correct")

        if not (0.0 < c1 < 0.5):
            raise ValueError('Invalid c1 value: {}'.format(c1))

        if not (c1 < c2 < 1.0):
            raise ValueError('Invalid c2 value: {}'.format(c2))

        if lr < 0.0:
            raise ValueError('Invalid lr value: {}'.format(lr))

        if not (0.0 < rho < 1.0):
            raise ValueError('Invalid rho value: {}'.format(rho))

        if not (0.0 < eta < 1.0):
            raise ValueError('Invalid eta value: {}'.format(eta))
        
        if max_ls%1!=0 or max_ls <= 0:
            raise ValueError('Invalid max_ls value: {}'.format(max_ls))

        defaults = dict(
            eps=eps,
            line_search=line_search,
            c1 = c1,
            c2 = c2,
            lr = lr,
            rho = rho,
            eta = eta,
            max_ls = max_ls
        )

        super(FR, self).__init__(params, defaults)

    def _get_A(p, d_p):
        A = torch.stack(
                        [torch.autograd.grad(
                            d_p[i],
                            p, 
                            grad_outputs=torch.ones_like(d_p[i]),
                            retain_graph=True)[0]
                        for i in range(0, len(d_p))])
        
        return A

    def Exact(A, d_p, d):
        rdotr = torch.dot(-d, d_p.data)

        z = torch.matmul(A, d)

        alpha = rdotr / torch.matmul(d, z)

        return alpha

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
                    state['g'] = copy.deepcopy(d_p.data)

                    # Negative grade of loss
                    state['d'] = copy.deepcopy(-d_p.data)

                    # Determine whether to calculate A
                    state['index'] = True
                else:
                    # Parameters that make gradient steps
                    state['beta'] = torch.norm(d_p.data) / torch.norm(state['g'])

                    state['g'] = copy.deepcopy(d_p.data)
                    
                    state['d'] = -state['g'] + state['beta'] * state['d']

                    state['index'] = False

                line_search = group['line_search']

                if line_search == 'None':
                    if state['index']:
                        state['A'] = FR._get_A(p, d_p)
                        alpha = FR.Exact(state['A'], d_p, state['d'])
                    else:
                        alpha = FR.Exact(state['A'], d_p, state['d'])

                elif line_search == 'Strong_Wolfe':
                    alpha = Strong_Wolfe()

                elif line_search == 'General_Wolfe':
                    alpha = General_Wolfe()
                
                elif line_search == 'General_Armijo':
                    alpha = General_Armijo()

                p.data.add_(state['d'], alpha=alpha)

        return loss