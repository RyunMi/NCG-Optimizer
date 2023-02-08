# failed Line Search Method :(
# Welcome to correct them!

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
            if abs(torch.dot(d_p.reshape(-1), d.reshape(-1))) <= -c2 * torch.dot(grad.reshape(-1), d.reshape(-1)):
            #if abs(float(torch.norm(d_p.reshape(-1)))) <= -c2 * float(torch.norm(grad.reshape(-1))):
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
        
def Strong_Wolfe3(func, x, d, c1, c2, iter):
    alpha = 1
    
    alpha1 = 0
    alpha2 = pow(10, 5)
    
    f1 = float(func())
    
    grad = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
    
    g1 = torch.dot(grad, d)
    g2 = -1

    for _ in range(iter):
        x.data = x.data + alpha * d
        lr = alpha
        
        if float(func()) > f1 + alpha * c1 * torch.dot(grad.reshape(-1), d.reshape(-1)):
            alpha2 = alpha
            f2 = float(func())

            alpha = alpha1 + 0.5 * (alpha2 - alpha1) / (1 + (f1 - f2) / ((alpha2 - alpha1) * g1))

            x.data = x.data - lr * d
            
            continue
        else:
            d_p = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
            
            if abs(torch.dot(d_p.reshape(-1), d.reshape(-1))) > -c2 * torch.dot(grad.reshape(-1), d.reshape(-1)):
                if torch.dot(d_p.reshape(-1), d.reshape(-1)) > 0:
                    alpha2 = alpha
                    f2 = float(func())
                    g2 = torch.dot(d_p.reshape(-1), d.reshape(-1))

                    beta = 2 * g1 + g2 - 3 * (f2 - f1) / (alpha2 - alpha1)
                    alpha = alpha1 - g1 * (alpha2 - alpha1) / (pow(pow(beta - g1, 2) - g1 * g2, 0.5) - beta)

                    x.data = x.data - lr * d

                    continue
                else:
                    if alpha2 != pow(10, 5):
                        alpha1 = alpha
                        f1 = float(func())
                        g1 = torch.dot(d_p.reshape(-1), d.reshape(-1))

                        if g2 > 0:
                            beta = 2 * g1 + g2 - 3 * (f2 - f1) / (alpha2 - alpha1)
                            alpha = alpha1 - g1 * (alpha2 - alpha1) / (pow(pow(beta - g1, 2) - g1 * g2, 0.5) - beta)
                        else:
                            alpha = alpha1 + 0.5 * (alpha2 - alpha1) / (1 + (f1 - f2) / ((alpha2 - alpha1) * g1))
                        
                        x.data = x.data - lr * d

                        continue
                    else:
                        bhat = 2 * torch.dot(d_p.reshape(-1), d.reshape(-1)) + g1 - 3 * (f1 - float(func())) / (alpha1 - alpha)
                        ahat = alpha - torch.dot(d_p.reshape(-1), d.reshape(-1)) * (alpha - alpha1) \
                                / (pow(pow(bhat - torch.dot(d_p.reshape(-1), d.reshape(-1)), 2) - \
                               torch.dot(d_p.reshape(-1), d.reshape(-1)) * g1, 0.5) + bhat)
                        
                        alpha1 = alpha
                        f1 = float(func())
                        g1 = torch.dot(d_p.reshape(-1), d.reshape(-1))
                        
                        alpha = ahat

                        x.data = x.data - lr * d

                        continue
            else:
                return alpha
    
    return alpha

def Wolfe(func, 
          x, 
          d, 
          lr, 
          c1, 
          c2, 
          eta, 
          k, 
          iter):
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
    
    alpha = 2 * lr

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