import torch

from torch.optim.optimizer import Optimizer

from Line_Search import Armijo
from Line_Search import Strong_Wolfe
from Line_Search import General_Wolfe

import copy

__all__ = ('FR',)

class FR(Optimizer):
    r"""Implements Fletcher-Reeves Conjugate Gradient.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        eps: term added to the denominator to improve
            numerical stability (default: 1e-5)
        line_search: designates line search to use (default: 'Armijo')
            Options:
                'None': uses exact line search(requires the loss is quadratic)
                'Armijo': uses Armijo backtracking line search
                'Strong_Wolfe': uses Strong_Wolfe bracketing line search
                'General_Wolfe': uses General_Wolfe bracketing line search
    
    Example:
         >>> import ncg_optimizer as optim
        >>> optimizer = optim.FR(model.parameters(), eps = 1e-5, line_search = 'Armijo',)
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
        line_search='Armijo'
    ):
        if eps < 0.0:
                raise ValueError('Invalid epsilon value: {}'.format(eps))

        if line_search not in [ 
            'Armijo', 
            'Strong_Wolfe', 
            'General_Wolfe',
            'None',
            ]:
            raise ValueError("Invalid line search: {}".format(line_search))
        
        defaults = dict(
            eps=eps,
            line_search=line_search,
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
                    state['index'] == 1
                else:
                    state['beta'] = torch.norm(d_p.data) / torch.norm(state['g'])

                    state['g'] = copy.deepcopy(d_p.data)
                    
                    state['d'] = -state['g'] + state['beta'] * state['d']

                    state['index'] == 0

                line_search = group['line_search']

                if line_search == 'None':
                    if state['index'] == 1:
                        state['A'] = FR._get_A(p, d_p)
                        alpha = FR.Exact(state['A'], d_p, state['d'])
                    else:
                        alpha = FR.Exact(state['A'], d_p, state['d'])

                elif line_search == 'Armijo':
                    alpha = Armijo()

                elif line_search == 'Strong_Wolfe':
                    alpha = Strong_Wolfe()

                elif line_search == 'General_Wolfe':
                    alpha = General_Wolfe()

                p = p.data.add_(state['d'], alpha=alpha)

        return loss