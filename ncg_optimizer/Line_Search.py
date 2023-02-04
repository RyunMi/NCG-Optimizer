import torch

def Armijo(func, x, g, d, lr, rho, c1, iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    g: grad.data of x
    d: data of direction vector
    lr: initialized stepsize
    rho: contraction factor
    c1: sufficient decrease constant
    iter: maximum step permitted
    """
    for _ in range(iter):
        F_o = float(func())

        x.data = x.data + lr * d

        if not float(func()) <= float(F_o + c1 * lr * torch.dot(g.reshape(-1), d.reshape(-1))):
            alpha = lr
            lr = lr * rho
        else:
            alpha = lr
            x.data = x.data - alpha * d
            break    
        
        x.data = x.data - alpha * d     

    return alpha

def Wolfe(func, x, d, lr, c1, c2, eta, k, iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    d: data of direction vector
    lr: initialized stepsize
    c1: sufficient decrease constant
    c2: curvature condition constant
    eta: adjustment factor
    k: step of Conjugate Gradient
    iter: maximum step permitted
    """
    F_o = float(func())
    
    x.data = x.data + 0.1 * lr * d

    grad = float(torch.norm(torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]))

    if float(func()) <= F_o and not (float(func()) - F_o - 0.1 * lr * grad):
        alpha = -grad * 0.01 * pow(lr, 2) / (2 * (float(func()) - F_o - 0.1 * lr * grad))
    else:
        alpha = 2 * lr

    x.data = x.data - 0.1 * lr * d

    a = 0
    b = pow(10, 5)

    beta_a = F_o
    beta_aa = torch.dot(x.grad.reshape(-1), d.reshape(-1))

    t1 = 1
    t2 = 0.1

    m = 1e-6 * abs(F_o)
    
    if k == 0:
        n = pow(10, 5)
    else:
        n = c1 * alpha * float(torch.norm(x.grad)) + 1 / pow(k,2)
    
    for _ in range(iter):
        r = alpha
        x.data = x.data + r * d
        
        d_p = float(torch.norm(torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]))
        
        if not (float(func()) <= F_o + min(m, n)):
            b = alpha
            beta_b = float(func())
            star = -beta_aa * pow(alpha, 2) / (2 * (beta_b - beta_a - alpha * beta_aa))
            t1 = 0.1 * t1
            alpha = min(max(star, a + t1 * (b - a)), b - t2 * (b - a))
            x.data = x.data - r * d
            continue
        else:
            if not (d_p >= c2 * torch.norm(x.grad)):
                t1 = 0.1
                t2 = 0.1 * t2
                if b == pow(10, 5):
                    a = alpha
                    beta_a = float(func())
                    beta_aa = d_p
                    alpha = eta * alpha
                    x.data = x.data - r * d
                    continue
                else:
                    a = alpha
                    beta_a = float(func())
                    beta_aa = d_p
                    star = -beta_aa * pow(alpha, 2) / (2 * (beta_b - beta_a - alpha * beta_aa))
                    alpha = min(max(star, a + t1 * (b - a)), b - t2 * (b - a))
                    x.data = x.data - r * d
                    continue
            else:
                return float(alpha)

    return float(alpha)