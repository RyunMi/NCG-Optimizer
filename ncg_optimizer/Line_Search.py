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
        
        if not float(func()) <= F_o + c1 * lr * g * d:
            alpha = lr
            lr = lr * rho
        else:
            alpha = lr
            break    
        
        x.data = x.data - alpha * d     

    return alpha

def Wolfe(func, x, d_p, d, lr, c1, c2, eta, k, iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    d_p: grad of x
    d: data of direction vector
    lr: initialized stepsize
    c1: sufficient decrease constant
    c2: curvature condition constant
    eta: adjustment factor
    k: step of Conjugate Gradient
    iter: maximum step permitted
    """
    F_o = float(func())

    grad = float(d_p.data)

    x.data = x.data + 0.1 * lr * d

    if float(func()) <= F_o:
        alpha = -grad * 0.01 * pow(lr, 2) / (2 * (float(func()) - F_o - 0.1 * lr * grad))
    else:
        alpha = 2 * lr

    x.data = x.data - 0.1 * lr * d

    a = 0
    b = pow(10, 5)

    beta_a = F_o
    beta_aa = grad * d

    t1 = 1
    t2 = 0.1

    m = 1e-6 * abs(F_o)
    
    if k == 0:
        n = pow(10, 5)
    else:
        n = c1 * alpha * grad + 1 / pow(k,2)
    
    for _ in range(iter):
        r = alpha
        x.data = x.data + r * d

        if not (float(func()) <= F_o + min(m, n)):
            b = alpha
            beta_b = float(func())
            star = -beta_aa * pow(alpha, 2) / (2 * (beta_b - beta_a - alpha * beta_aa))
            t1 = 0.1 * t1
            alpha = min(max(star, a + t1 * (b - a)), b - t2 * (b - a))
            x.data = x.data - r * d
            continue
        else:
            if not (float(d_p.data) >= c2 * grad):
                t1 = 0.1
                t2 = 0.1 * t2
                if b == pow(10, 5):
                    a = alpha
                    beta_a = float(func())
                    beta_aa = float(d_p.data)
                    alpha = eta * alpha
                    x.data = x.data - r * d
                    continue
                else:
                    a = alpha
                    beta_a = float(func())
                    beta_aa = float(d_p.data)
                    star = -beta_aa * pow(alpha, 2) / (2 * (beta_b - beta_a - alpha * beta_aa))
                    alpha = min(max(star, a + t1 * (b - a)), b - t2 * (b - a))
                    x.data = x.data - r * d
                    continue
            else:
                return alpha
    
    return alpha

def Strong_Wolfe(func, x, d_p, d, lr, c1, c2, amax, iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    d_p: grad of x
    d: data of direction vector
    lr: initialized stepsize
    c1: sufficient decrease constant
    c2: curvature condition constant
    amax: maximum step length
    iter: maximum step permitted
    """
    F_o = float(func())

    grad = float(d_p.data)

    a0 = lr
    a1 = (0 + amax) / 2

    i = 1

    for _ in range(iter):
        x.data = x.data + a0 * d
        F_n = float(func())
        x.data = x.data - a0 * d

        x.data = x.data + a1 * d

        if float(func()) > F_o + a1 * c1 * grad or (float(func()) >= F_n and i > 1):
            x.data = x.data - a1 * d
            alpha = zoom(func, x, d_p, d, c1, c2, a0, a1, iter)
            x.data = x.data + a1 * d
            return alpha
        else:
            if abs(float(d_p.data) * d) <= -c2 * grad * d:
                alpha = a1
                return alpha
            else:
                if float(d_p.data) * d >= 0:
                    x.data = x.data - a1 * d
                    alpha = zoom(func, x, d_p, d, c1, c2, a0, a1, iter)
                    x.data = x.data + a1 * d
                    return alpha
                else:
                    x.data = x.data - a1 * d
                    a0 = a1
                    a1 = (a1 + amax) / 2
                    i = i + 1
    
    return alpha

def zoom(f, x, p, d, c1, c2, a0, a1, t):
    """
    zoom: takes in two values a0 and a1 that bounds the interval [a0,a1] 
    containing the step lengths that satisfy the strong Wolfe conditions.
    """
    beta = (a0 + a1) / 2
    
    F_o = f()
    grad = float(p.data)

    for _ in range(round(t/2)):
        x.data = x.data + a0 * d
        F_n = float(f())
        x.data = x.data - a0 * d

        x.data = x.data + beta * d
        
        if float(f()) > F_o + beta * c1 * grad or float(f()) >= F_n:
            a1 = beta
            x.data = x.data - beta * d
            continue
        else:
            if abs(float(p.data) * d) <= -c2 * grad * d:
                alpha = beta
                return alpha
            else:
                if (float(p.data) * d) * (a1 - a0) >= 0:
                    a1 = a0
                a0 = beta
        x.data = x.data - beta * d

    return alpha