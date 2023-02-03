import numpy as np

import torch
from torch import nn

import ncg_optimizer as optim

MyDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
def main():
    torch.manual_seed(42)
    x_data, y_data = make_dataset()
    x_data = x_data.to(MyDevice)
    y_data = y_data.to(MyDevice)
    model = LogisticRegression().to(MyDevice)
    iterations = 500
    loss_fn = nn.BCELoss()
    #optimizer = optim.FR(model.parameters(), eps=1e-3, line_search='Armijo', lr=0.1)
    optimizer = optim.FR(model.parameters(), eps=1e-3, line_search='Wolfe', c2=0.5, lr=0.5, eta=5)
    for _ in range(iterations):
        def closure():
            optimizer.zero_grad()
            y_pred = model(x_data)
            loss = loss_fn(y_pred, y_data)
            print(loss)
            loss.backward(create_graph=True)
            return loss
        optimizer.step(closure)

if __name__ == '__main__':
    main()