import torch

import math

#import ncg_optimizer as optim
import sys
sys.path.append('..\\ncg_optimizer')
from prp import PRP
from dy import DY
from hs import HS
MyDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def quadratic(tensor):
#     x, y = tensor
#     a = 1.0
#     b = 1.0
#     return (x ** 2) / a + (y ** 2) / b + 2 * x * y - x - y
# initial_state = (-2.0, -2.0)
# min_loc = (0.25, 0.25)
# min = torch.tensor(quadratic(min_loc),device=MyDevice)

def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 1 * (y - x ** 2) ** 2
initial_state = (-2.0, -2.0)
min_loc = (1, 1)
min = torch.tensor(rosenbrock(min_loc),device=MyDevice)

# def beale(tensor):
#     x, y = tensor
#     f = (
#         (1.5 - x + x * y) ** 2
#         + (2.25 - x + x * y ** 2) ** 2
#         + (2.625 - x + x * y ** 3) ** 2
#     )
#     return f
# initial_state = (1.5, 1.5)
# min_loc = (3, 0.5)
# min = torch.tensor(beale(min_loc),device=MyDevice)

# def rastrigin(tensor, lib=torch):
#     x, y = tensor
#     A = 10
#     f = (
#         A * 2
#         + (x ** 2 - A * lib.cos(x * math.pi * 2))
#         + (y ** 2 - A * lib.cos(y * math.pi * 2))
#     )
#     return f
# initial_state = (-2.0, 3.5)
# min_loc = (0, 0)
# min = torch.tensor(rastrigin(torch.tensor(min_loc,device=MyDevice)))

x = torch.tensor(initial_state,device=MyDevice).requires_grad_(True)
x_min = torch.tensor(min_loc,device=MyDevice)
iterations = 500

def main():
    #optimizer = optim.LCG([x], eps = 1e-3)
    #optimizer = optim.FR([x], eps=1e-3, line_search='Armijo', lr=0.1)
    #optimizer = optim.FR([x], eps=1e-3, line_search='Wolfe', c2=0.9, lr=0.5, eta=5)
    #optimizer = FR([x], eps=1e-3, line_search='Armijo', lr=0.1)
    optimizer = DY([x], eps=1e-3, line_search='Wolfe', c2=0.4, lr=0.5, eta=5)
    #optimizer = FR([x], eps=1e-3, line_search='None')
    for _ in range(iterations+500):
        def closure():
            #print(x)
            optimizer.zero_grad()
            #f = quadratic(x)
            f = rosenbrock(x)
            #f = beale(x)
            #f = rastrigin(x)
            f.backward(retain_graph=True, create_graph=True)
            return f
        print(closure() - min)
        optimizer.step(closure)
        if torch.allclose(x, x_min.float(), atol=0.01) or torch.allclose(closure(), min.float(), atol=0.01):
            break
    torch.allclose(closure(), min.float(), atol=0.05)

if __name__ == '__main__':
    main()