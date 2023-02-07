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

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.linear2(output)
        y_pred = torch.sigmoid(output)
        return y_pred


def ids(v):
    return '{} {}'.format(v[0].__name__, v[1:])


optimizers = [
    # (optim.FR,{'line_search': 'Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.PRP,{'line_search': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.HS,{'line_search': 'Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.CD,{'line_search': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.LS,{'line_search': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.DY,{'line_search': 'Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.HZ,{'line_search': 'Wolfe','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.HS_DY,{'line_search': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'lr': 0.5, 'eta': 5}, 500),
    # (optim.BASIC,{'method': 'FR', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    # (optim.BASIC,{'method': 'PRP', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    #(optim.BASIC,{'method': 'HS', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    # (optim.BASIC,{'method': 'CD', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    # (optim.BASIC,{'method': 'DY', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    # (optim.BASIC,{'method': 'LS', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    (optim.BASIC,{'method': 'HZ', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
    # (optim.BASIC,{'method': 'HS-DY', 'line_search': 'Armijo','c1': 1e-4,'c2': 0.9, 'lr': 0.5, 'rho': 0.5}, 500),
]


@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_basic_nn_modeloptimizer_config(optimizer_config):
    torch.manual_seed(42)
    x_data, y_data = make_dataset()
    model = LogisticRegression()

    loss_fn = nn.BCELoss()
    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)
    #init_loss = None
    for _ in range(iterations):
        def closure():
            optimizer.zero_grad()
            y_pred = model(x_data)
            #print(y_pred)
            loss = loss_fn(y_pred, y_data)
            loss.backward(create_graph=True)
            return loss
        # if init_loss is None:
        #     init_loss = loss
        optimizer.step(closure)
        print(closure())
    #assert init_loss.item() > 2.0 * loss.item()
    assert closure() < 0.2