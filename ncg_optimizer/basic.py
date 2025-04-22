
import torch

from torch.optim.optimizer import Optimizer

from ncg_optimizer.Line_Search import Armijo
from ncg_optimizer.Line_Search import Curvature
from ncg_optimizer.Line_Search import Strong_Wolfe
from ncg_optimizer.Line_Search import Weak_Wolfe
from ncg_optimizer.Line_Search import Goldstein

import copy

import warnings

__all__ = ('BASIC',)

class BASIC(Optimizer):
    """
    Implements Basic Nonlinear Conjugate Gradient.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        
        eps: Parameters controlling iteration precision (default: 1e-3)
       
         method: select basic congjugate gradient (default: 'PRP')
            Optios:
                'FR': Implements Fletcher-Reeves Conjugate Gradient
                'PRP': Implements Polak Ribiere Polyak Conjugate Gradient.
                'HS': Implements Hestenes-Stiefel Conjugate Gradient.
                'CD': Implements Conjugate Descent Method.
                'DY': Implements Dai-Yuan Conjugate Gradient.
                'LS': Implements Liu-Storey Conjugate Gradient.
                'HZ': Implements Hager-Zhang Conjugate Gradient.
                'HS-DY': Implements Hybird Hestenes-Stiefel + Dai-Yuan Conjugate Gradient.
        
        line_search: designates line search to use (default: 'Armijo')
            Options:
                'None': uses exact line search(requires the loss is quadratic)
                'Armijo': uses Armijo line search
                'Curvature': uses Curvature line search
                'Strong Wolfe': uses Strong Wolfe line search
                'Weak Wolfe': uses Weak Wolfe line search
                'Goldstein': uses Goldstein line search

        c1: armijo or sufficient decrease constant in (0, 1) (default: 1e-4)
        
        c2: curvature condition constant in (c1, 1) (default: 0.1)

        c: goldstein condition constant in (0, 0.5) (default: 0.35)
        
        lr: initial step length of Line Search (default: 1)
        
        rho: contraction factor of Backtracking Line Search (default: 0.5)

        max_ls: maximum number of line search steps permitted (default: 10)
    
    Example:
        >>> import ncg_optimizer as optim
        >>> optimizer = optim.BASIC(
        >>>     model.parameters(), eps = 1e-3, method = 'PRP',
        >>>     line_search = 'Armijo', c1 = 1e-4, c2 = 0.1,
        >>>     lr = 1, rho = 0.5, max_ls = 10)
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
        method = 'PRP',
        line_search = 'Armijo',
        c1 = 1e-4,
        c2 = 0.1,
        c = 0.35,
        lr = 1,
        rho = 0.5,
        max_ls = 10,
    ):
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))

        if method not in [
            'FR',
            'PRP', 
            'HS',
            'CD',
            'DY',
            'LS',
            'HZ',
            'HS-DY',
            ]:
            raise ValueError('Invalid method: {}'.format(method))

        if line_search not in [
            'Armijo',
            'Curvature',
            'Strong_Wolfe',
            'Weak_Wolfe',
            'Goldstein',
            'None',
            ]:
            raise ValueError('Invalid line search: {}'.format(line_search))
        elif line_search == 'None':
            warnings.warn("Unless loss is a quadratic function, this is not correct")

        if not (0.0 < c1 < 1.0):
            raise ValueError('Invalid c1 value: {}'.format(c1))

        if not (c1 < c2 < 1.0):
            raise ValueError('Invalid c2 value: {}'.format(c2))

        if not (0.0 < c < 0.5):
            raise ValueError('Invalid c value: {}'.format(c))

        if lr < 0.0:
            raise ValueError('Invalid lr value: {}'.format(lr))

        if not (0.0 < rho < 1.0):
            raise ValueError('Invalid rho value: {}'.format(rho))

        if max_ls % 1 != 0 or max_ls <= 0:
            raise ValueError('Invalid max_ls value: {}'.format(max_ls))

        defaults = dict(
            eps=eps,
            method = method,
            line_search=line_search,
            c1 = c1,
            c2 = c2,
            c = c,
            lr = lr,
            rho = rho,
            max_ls = max_ls,
        )

        super(BASIC, self).__init__(params, defaults)

    def _get_A(p, d_p):

        A = torch.stack(
                        [torch.autograd.grad(
                            d_p[i],
                            p, 
                            grad_outputs=torch.ones_like(d_p[i]),
                            retain_graph=True)[0]
                        for i in range(0, len(d_p))])
        
        return A

    # Find the optimal learning rate by performing Exact Line Search
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

                method = group['method']
                line_search = group['line_search']
                c1 = group['c1']
                c2 = group['c2']
                c = group['c']
                lr = group['lr']
                rho = group['rho']
                eps = group['eps']
                max_ls = group['max_ls']

                if len(state) == 0:
                    # Grade of quadratic functions
                    state['g'] = copy.deepcopy(d_p.data)

                    if torch.norm(state['g']) < group['eps']:
                        # Stop condition
                        return loss

                    # Direction vector
                    state['d'] = copy.deepcopy(-d_p.data)

                    # Determine whether to calculate A
                    state['index'] = True
                    
                    # Step of Conjugate Gradient
                    state['step'] = 0

                    # initialized step length
                    state['alpha'] = lr
                else:
                    # 1. Calculate the conugate gradient coefficient "beta" by using the method of your choice.
                    # Parameters that make gradient steps
                    if method == 'FR':
                        state['beta'] = torch.norm(d_p.data) / torch.norm(state['g'])
                    
                    elif method == 'PRP':
                        state['beta'] = torch.dot(
                            d_p.data.reshape(-1), 
                            (d_p.data.reshape(-1) - state['g'].reshape(-1))) / torch.norm(state['g'])
                    
                    elif method == 'HS':
                        state['beta'] = torch.dot(
                            d_p.data.reshape(-1), 
                            (d_p.data.reshape(-1) - state['g'].reshape(-1))) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                (d_p.data.reshape(-1) - state['g'].reshape(-1)))
                    
                    elif method == 'CD':
                        state['beta'] = -torch.norm(d_p.data) \
                            / torch.dot(state['d'].data.reshape(-1), state['g'].reshape(-1))
                    
                    elif method == 'DY':
                        state['beta'] = torch.norm(d_p.data) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                (d_p.data.reshape(-1) - state['g'].reshape(-1)))
                    
                    elif method =='LS':
                        state['beta'] = -torch.dot(
                            d_p.data.reshape(-1), 
                            (d_p.data.reshape(-1) - state['g'].reshape(-1))) \
                            / torch.dot(state['d'].data.reshape(-1), state['g'].reshape(-1))
                    
                    elif method =='HZ':
                        Q = d_p.data - state['g']
                        M = Q - 2 *  torch.norm(Q) \
                            / torch.dot(state['d'].reshape(-1), Q.reshape(-1)) * state['d']
                        N = d_p.data / torch.dot(state['d'].reshape(-1), Q.reshape(-1))
                        state['beta'] = torch.dot(M.reshape(-1), N.reshape(-1))
                    
                    elif method =='HS-DY':
                        state['beta'] = max(0, 
                            min(torch.dot(d_p.data.reshape(-1), 
                                    (d_p.data.reshape(-1) - state['g'].reshape(-1))) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                    (d_p.data.reshape(-1) - state['g'].reshape(-1))),
                            torch.norm(d_p.data) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                    (d_p.data.reshape(-1) - state['g'].reshape(-1)))))
                    
                    state['g'] = copy.deepcopy(d_p.data)

                    if torch.norm(state['g']) < group['eps']:
                        return loss
                    
                    state['d'] = -state['g'] + state['beta'] * state['d']

                    state['index'] = False

                # 2. Calculate the optimal learning rate coefficient
                # 2.a If "None" was given as a line search method, then perform Exact Line Search
                if line_search == 'None':
                    if state['index']:
                        state['A'] = BASIC._get_A(p, d_p)
                        state['alpha'] = BASIC.Exact(state['A'], d_p, state['d'])
                    else:
                        state['alpha'] = BASIC.Exact(state['A'], d_p, state['d'])

                # 2.b Otherwise, perform Inexact Line Search by using the method of your choice.
                elif line_search == 'Armijo':
                    state['alpha'] = Armijo(closure, p, state['g'], state['d'], state['alpha'], rho, c1, max_ls)

                elif line_search == 'Curvature':
                    state['alpha'] = Curvature(closure, p, state['g'], state['d'], state['alpha'], rho, c2, max_ls)

                elif line_search == 'Strong_Wolfe':
                    state['alpha'] = Strong_Wolfe(closure, p, lr, state['d'], c1, c2, eps, max_ls)

                elif line_search == 'Weak_Wolfe':
                    state['alpha'] = Weak_Wolfe(closure, p, lr, state['d'], c1, c2, eps, max_ls)

                elif line_search == 'Goldstein':
                    state['alpha'] = Goldstein(closure, p, lr, state['d'], c, eps, max_ls)

                p.data.add_(state['d'], alpha=state['alpha'])

        return loss
