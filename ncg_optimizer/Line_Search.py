def Armijo():
    alpha = 42

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