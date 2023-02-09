import numpy as np

import pytest

import torch
from torch import nn

import ncg_optimizer as optim


def make_dataset(seed=42):
    rng = np.random.RandomState(seed)
    N = 100
    D = 2

    X = rng.randn(N, D) * 2

    # center the first N/2 points at (-2,-2)
    mid = N // 2
    X[:mid, :] = X[:mid, :] - 2 * np.ones((mid, D))

    # center the last N/2 points at (2, 2)
    X[mid:, :] = X[mid:, :] + 2 * np.ones((mid, D))

    # labels: first N/2 are 0, last N/2 are 1
    Y = np.array([0] * mid + [1] * mid).reshape(100, 1)

    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    return x, y


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)
        self.init_normal()

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.linear2(output)
        y_pred = torch.sigmoid(output)
        return y_pred
    
    def init_normal(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

net = LogisticRegression()

def ids(v):
    return '{} {}'.format(v[0].__name__, v[1:])


optimizers = [
    (optim.BASIC,{'method': 'FR', 'line_search': 'Strong_Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'PRP', 'line_search': 'Strong_Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5}, 500),
    (optim.BASIC,{'method': 'HS', 'line_search': 'Strong_Wolfe','c1': 1e-4,'c2': 0.3, 'lr': 0.5}, 500),
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
    (optim.BASIC,{'method': 'LS', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'HZ', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'HS-DY', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
]


@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_basic_nn_modeloptimizer_config(optimizer_config):
    torch.manual_seed(42)
    
    x_data, y_data = make_dataset()
    
    model = LogisticRegression()
    #loss_fn = nn.BCELoss()
    
    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)
 
    for _ in range(iterations):
        def closure():
            optimizer.zero_grad()
            
            y_pred = model(x_data)
            y_pred = torch.clamp(y_pred, min=1e-6, max=1-1e-6)

            #loss = loss_fn(y_pred, y_data)
            loss = - y_data * torch.log(y_pred) - (1-y_data) * torch.log(1-y_pred)
            loss = loss.mean()
            
            loss.backward(create_graph=True)
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            return loss

        optimizer.step(closure)

        # if closure().item() < 0.15:
        #     break
    
    assert closure().item() < 0.15