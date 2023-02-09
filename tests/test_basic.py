import torch

import pytest

import math

import ncg_optimizer as optim

MyDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def quadratic(tensor):
    x, y = tensor
    a = 1.0
    b = 1.0
    return (x ** 2) / a + (y ** 2) / b 

def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 1 * (y - x ** 2) ** 2

def beale(tensor):
    x, y = tensor
    f = (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y ** 2) ** 2
        + (2.625 - x + x * y ** 3) ** 2
    )
    return f

def rastrigin(tensor, lib=torch):
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x ** 2 - A * lib.cos(x * math.pi * 2))
        + (y ** 2 - A * lib.cos(y * math.pi * 2))
    )
    return f

cases = [
    (quadratic, (-2.0, -2.0), (0, 0)),
    (rosenbrock, (-2.0, -2.0), (1, 1)),
    (beale, (1.5, 1.5), (3, 0.5)),
    (rastrigin, (-2.0, 3.5), (0, 0)),
]

def ids(v):
    n = '{} {}'.format(v[0].__name__, v[1:])
    return n

optimizers = [
    (optim.BASIC,{'method': 'FR', 'line_search': 'Strong_Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'PRP', 'line_search': 'Strong_Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'HS', 'line_search': 'Strong_Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'CD', 'line_search': 'Strong_Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'DY', 'line_search': 'Strong_Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'LS', 'line_search': 'Strong_Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'HZ', 'line_search': 'Strong_Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'HS-DY', 'line_search': 'Strong_Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'FR', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'PRP', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'HS', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'CD', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'DY', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'LS', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 1, 'rho': 0.8}, 500),
    (optim.BASIC,{'method': 'HZ', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'HS-DY', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
]

@pytest.mark.parametrize('case', cases, ids=ids)
@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_benchmark_function(case, optimizer_config):
    func, initial_state, min_loc = case
    optimizer_class, config, iterations = optimizer_config
    
    x = torch.tensor(initial_state).requires_grad_(True)
    x_min = torch.tensor(min_loc)
    optimizer = optimizer_class([x], **config)
    
    for _ in range(iterations):
        def closure():
            optimizer.zero_grad()
            
            f = func(x)
            f.backward(retain_graph=True, create_graph=True)
            
            return f

        optimizer.step(closure)

        # if torch.allclose(x, x_min.float(), atol=0.01) or torch.allclose(closure(), func(x_min).float(), atol=0.05):
        #     break

    assert torch.allclose(closure(), func(x_min).float(), atol=0.05)

    name = optimizer.__class__.__name__
    assert name in optimizer.__repr__()