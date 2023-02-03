import torch

def Strong_Wolfe1(func, x, d, lr, c1, c2, eta, iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    d: data of direction vector
    lr: initialized stepsize
    c1: sufficient decrease constant
    c2: curvature condition constant
    eta: adjustment factor
    iter: maximum step permitted
    """
    F_o = float(func())

    a0 = 0
    a1 = lr

    i = 1

    grad = x.grad
    
    for _ in range(iter):
        x.data = x.data + a0 * d
        F_n = float(func())
        x.data = x.data - a0 * d

        x.data = x.data + a1 * d
        d_p = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]

        if float(func()) > F_o + a1 * c1 * torch.dot(grad.reshape(-1), d.reshape(-1)) or (float(func()) >= F_n and i > 1):
            
            #beta = a0 + (a1 - a0) / 2 * (1 + (F_n - float(func())) / ((a1 - a0) * torch.norm(d_p.reshape(-1))))
            beta = (a0 + a1) / 2
            x.data = x.data - a1 * d
            
            f = float(func())
            pd = x.grad
            
            for _ in range(iter):
                
                x.data = x.data + a0 * d
                g = float(func())
                x.data = x.data - a0 * d + beta * d

                p = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
                
                alpha, a0, a1, index = zoom(func(), pd, d, c1, c2, a0, a1, beta, f, g, p)

                x.data = x.data - beta * d

                if index:
                    break
            
            return float(alpha)
        else:
            if abs(torch.dot(d_p.reshape(-1), d.reshape(-1))) <= -c2 * torch.dot(grad.reshape(-1), d.reshape(-1)):
                alpha = a1
                return float(alpha)
            else:
                if torch.dot(d_p.reshape(-1), d.reshape(-1)) >= 0:
                    
                    x.data = x.data - a1 * d
                    x.data = x.data + a0 * d
                    gg = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
                    x.data = x.data - a0 * d
                    #beta = a1 + (a1 - a0) * torch.norm(d_p.reshape(-1)) / (gg - torch.norm(d_p.reshape(-1)))
                    beta = (a0 + a1) / 2
                    f = float(func())
                    pd = x.grad
                    
                    for _ in range(iter):
                    
                        x.data = x.data + a1 * d
                        g = float(func())
                        x.data = x.data - a1 * d + beta * d
                    
                        p = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
                    
                        alpha, a1, a0, index = zoom(func(), pd, d, c1, c2, a1, a0, beta, f, g, p)
                    
                        x.data = x.data - beta * d

                        if index:
                            break
                    
                    return float(alpha)
                else:
                    x.data = x.data - a1 * d
                    a0 = a1
                    a1 = a1 * eta
                    i = i + 1
    
    return float(a1)

def zoom(f, grad, d, c1, c2, a0, a1, beta, F_o, F_n, p):
    """
    zoom: takes in two values a0 and a1 that bounds the interval [a0,a1] 
    containing the step lengths that satisfy the strong Wolfe conditions.
    """
    index = False
    if float(f) > F_o + beta * c1 * torch.dot(grad.reshape(-1), d.reshape(-1)) or float(f) >= F_n:
        a1 = beta
    else:
        if abs(torch.dot(p.reshape(-1), d.reshape(-1))) <= -c2 * torch.dot(grad.reshape(-1), d.reshape(-1)):
            alpha = beta
            index = True
            return alpha, a0 ,a1, index
        else:
            if (torch.dot(p.reshape(-1), d.reshape(-1))) * (a1 - a0) >= 0:
                a1 = a0
        a0 = a0
    
    return beta, a0, a1, index

def Strong_Wolfe2(func, x, d, lr, c1, c2, iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    d: data of direction vector
    lr: initialized stepsize
    c1: sufficient decrease constant
    c2: curvature condition constant
    iter: maximum step permitted
    """
    amin = 0
    amax = pow(10,5)
    F_o = float(func())
    
    x.data = x.data + 0.1 * lr * d

    grad = float(torch.norm(torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]))
    
    if float(func()) <= F_o and not (float(func()) - F_o - 0.1 * lr * grad):
        alpha = -grad * 0.01 * pow(lr, 2) / (2 * (float(func()) - F_o - 0.1 * lr * grad))
    else:
        alpha = 2 * lr

    x.data = x.data - 0.1 * lr * d
    grad = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
    i = 1
    while True:
        x.data = x.data + alpha * d
        d_p = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
        if float(func()) > F_o + alpha * c1 * torch.dot(grad.reshape(-1), d.reshape(-1)):
            x.data = x.data - alpha * d
            amax = alpha
            alpha = (amax + amin) / 2
        else:
            if abs(torch.dot(d_p.reshape(-1), d.reshape(-1))) <= c2 * abs(torch.dot(grad.reshape(-1), d.reshape(-1))):
                return alpha
            amin = alpha
            x.data = x.data - alpha * d
            if amax == pow(10,5):
                alpha = 2 * alpha
            elif amax < pow(10,5):
                alpha = (amax + amin) / 2
        i = i + 1
        if i == iter:
            return alpha