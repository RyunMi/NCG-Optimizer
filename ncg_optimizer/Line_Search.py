def Armijo(func, x, g, d, lr, rho, c1, iter):
    for _ in range(iter):
        F_o = float(func())

        x.data = x.data + lr*d
        
        if not float(func()) <= F_o + c1*lr*g*d:
            alpha = lr
        
            lr = lr*rho
        else:
            alpha = lr
        
            break    
        
        x.data = x.data - alpha*d     

    return alpha

def Wolfe():
    alpha = 42
    
    return alpha

def Strong_Wolfe(func, p):
    F_new = func()
    p.data = p.data + 1
    print(F_new)
    p.data = p.data - 1
    alpha = 42
    
    return alpha

def General_Wolfe():
    alpha = 42
    
    return alpha

def General_Armijo():
    alpha = 42

    return alpha