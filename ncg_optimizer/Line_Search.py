
import torch


###################################################################################################### Armijo Condition function
def Armijo(func, 
           x, 
           g, 
           d, 
           lr, 
           rho, 
           c1, 
           iter):
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
        # Evaluate previous point
        f_prev = float(func())                                          # f(x_t)
        g_prev = g                                                      # grad[f(x_t)]
        gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))         # grad[f(x_t)] * d_t

        # Evaluate next point        
        x.data = x.data + lr * d                                        # x_t+1 = x_t + lr * d_t
        f_new = float(func())                                           # f(x_t+1) = f(x_t + lr * d_t)

        if not f_new <= float(f_prev + c1 * lr * gtd_prev):             # Armijo Condition
            alpha = lr
            lr = lr * rho
        else:
            alpha = lr
            x.data = x.data - alpha * d
            break
        
        x.data = x.data - alpha * d

    return alpha


###################################################################################################### Curvature Condition function
def Curvature(func, 
           x, 
           g, 
           d, 
           lr, 
           rho, 
           c2, 
           iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    g: grad.data of x
    d: data of direction vector
    lr: initialized stepsize
    rho: contraction factor
    c2: sufficient decrease constant
    iter: maximum step permitted
    """
    for _ in range(iter):
        # Evaluate previous point
        g_prev = g                                                                       # grad[f(x_t)]
        gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))                          # grad[f(x_t)] * d_t

        # Evaluate next point        
        x.data = x.data + lr * d                                                         # x_t+1 = x_t + lr * d_t
        g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]  # grad[f(x_t+1)] = grad[f(x_t + lr * d_t)]
        gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))                            # grad[f(x_t+1) * d_t] = grad[f(x_t + lr * d_t)] * d_t
        
        if not float(gtd_new) >= float(c2 * gtd_prev):                                   # Curvature Condition
            alpha = lr
            lr = lr * rho
        else:
            alpha = lr
            x.data = x.data - alpha * d
            break
        
        x.data = x.data - alpha * d

    return alpha


###################################################################################################### Cubic Interpolate function
def Cubic_Interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # Inspired by https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # cubic interpolation of 2 points
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


###################################################################################################### Strong Wolfe Conditions function
def Strong_Wolfe(func,
                  x,
                  t,
                  d,
                  c1,
                  c2,
                  tolerance_change=1e-3,
                  max_ls=10):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    t: initialized stepsize
    d: data of direction vector
    c1: sufficient decrease constant
    c2: curvature condition constant
    tolerance_change: min line-search bracket
    iter: maximum step permitted
    Inspired by https://github.com/torch/optim/blob/master/lswolfe.lua
    """
    
    d_norm = d.abs().max()
    t_prev, f_prev  = 0, float(func()) 
    g_prev = x.grad
    gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))
    
    # evaluate objective and gradient using initial step
    x.data = x.data + t * d
    f_new = float(func())
    g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
    gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
    x.data = x.data - t * d
    
    done = False
    ls_iter = 0

    # bracket an interval containing a point satisfying the Wolfe criteria
    while ls_iter < max_ls:
        # check conditions
        if (f_new > float(f_prev + c1 * t * gtd_prev)) or (ls_iter > 1 and f_new >= f_prev):         # Armijo Condition
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(float(gtd_new)) <= c2 * abs(float(gtd_prev)):                                         # Modified Curvature Condition
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = Cubic_Interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        x.data = x.data + t * d
        f_new = float(func())
        g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
        gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
        x.data = x.data - t * d
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [float(func()), f_new]
        bracket_g = [x.grad, g_new]

    # zoom phase: Now having a point satisfying the criteria, or
    # a bracket around it. Refining the bracket until finding the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = Cubic_Interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that there are making sufficient progress:
        # if there made insufficient progress in the last step,  or `t` is at one of the boundary,
        # `t` will moved to a position which is `0.1 * len(bracket)` away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate previous point
        f_prev = float(func()) 
        g_prev = x.grad
        gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))
        # Evaluate new point
        x.data = x.data + t * d
        f_new = float(func())
        g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
        gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
        x.data = x.data - t * d
        ls_iter += 1

        if (f_new > float(f_prev + c1 * t * gtd_prev)) or (f_new >= bracket_f[low_pos]):         # Armijo Condition
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(float(gtd_new)) <= c2 * abs(float(gtd_prev)):                                 # Modified Curvature Condition
                # Strong Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]

    return float(t)


###################################################################################################### Weak Wolfe Conditions function
def Weak_Wolfe(func,
                  x,
                  t,
                  d,
                  c1,
                  c2,
                  tolerance_change=1e-3,
                  max_ls=10):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    t: initialized stepsize
    d: data of direction vector
    c1: sufficient decrease constant
    c2: curvature condition constant
    tolerance_change: min line-search bracket
    iter: maximum step permitted
    Inspired by https://github.com/torch/optim/blob/master/lswolfe.lua
    """
    
    d_norm = d.abs().max()
    t_prev, f_prev  = 0, float(func()) 
    g_prev = x.grad
    gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))
    
    # evaluate objective and gradient using initial step
    x.data = x.data + t * d
    f_new = float(func())
    g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
    gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
    x.data = x.data - t * d
    
    done = False
    ls_iter = 0

    # bracket an interval containing a point satisfying the Wolfe criteria
    while ls_iter < max_ls:
        # check conditions
        if (f_new > float(f_prev + c1 * t * gtd_prev)) or (ls_iter > 1 and f_new >= f_prev):         # Armijo Condition
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if float(gtd_new) >= float(c2 * gtd_prev):                                                   # Curvature Condition
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = Cubic_Interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        x.data = x.data + t * d
        f_new = float(func())
        g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
        gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
        x.data = x.data - t * d
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [float(func()), f_new]
        bracket_g = [x.grad, g_new]

    # zoom phase: Now having a point satisfying the criteria, or
    # a bracket around it. Refining the bracket until finding the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = Cubic_Interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that there are making sufficient progress:
        # if there made insufficient progress in the last step,  or `t` is at one of the boundary,
        # `t` will moved to a position which is `0.1 * len(bracket)` away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate previous point
        f_prev = float(func()) 
        g_prev = x.grad
        gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))
        # Evaluate new point
        x.data = x.data + t * d
        f_new = float(func())
        g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
        gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
        x.data = x.data - t * d
        ls_iter += 1

        if (f_new > float(f_prev + c1 * t * gtd_prev)) or (f_new >= bracket_f[low_pos]):         # Armijo Condition
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if float(gtd_new) >= float(c2 * gtd_prev):                                           # Curvature Condition
                # Weak Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]

    return float(t)


###################################################################################################### Goldstein Conditions function
def Goldstein(func,
                  x,
                  t,
                  d,
                  c,
                  tolerance_change=1e-3,
                  max_ls=10):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    t: initialized stepsize
    d: data of direction vector
    c: sufficient decrease constant
    1-c: curvature condition constant
    tolerance_change: min line-search bracket
    iter: maximum step permitted
    Inspired by https://github.com/torch/optim/blob/master/lswolfe.lua
    """
    
    # Set the sufficient decrease and curvature constants
    c1 = c
    c2 = 1 - c
    
    d_norm = d.abs().max()
    t_prev, f_prev  = 0, float(func()) 
    g_prev = x.grad
    gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))
    
    # evaluate objective and gradient using initial step
    x.data = x.data + t * d
    f_new = float(func())
    g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
    gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
    x.data = x.data - t * d
    
    done = False
    ls_iter = 0

    # bracket an interval containing a point satisfying the Wolfe criteria
    while ls_iter < max_ls:
        # check conditions
        if (f_new > float(f_prev + c1 * t * gtd_prev)) or (ls_iter > 1 and f_new >= f_prev):         # Armijo Condition
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if f_new >= float(f_prev + c2 * t * gtd_prev):                                               # Modified Armijo Condition
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = Cubic_Interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        x.data = x.data + t * d
        f_new = float(func())
        g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
        gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
        x.data = x.data - t * d
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [float(func()), f_new]
        bracket_g = [x.grad, g_new]

    # zoom phase: Now having a point satisfying the criteria, or
    # a bracket around it. Refining the bracket until finding the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = Cubic_Interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that there are making sufficient progress:
        # if there made insufficient progress in the last step,  or `t` is at one of the boundary,
        # `t` will moved to a position which is `0.1 * len(bracket)` away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate previous point
        f_prev = float(func()) 
        g_prev = x.grad
        gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))
        # Evaluate new point
        x.data = x.data + t * d
        f_new = float(func())
        g_new = torch.autograd.grad(func(), x, retain_graph=True, create_graph=True)[0]
        gtd_new = torch.dot(g_new.reshape(-1), d.reshape(-1))
        x.data = x.data - t * d
        ls_iter += 1

        if (f_new > float(f_prev + c1 * t * gtd_prev)) or (f_new >= bracket_f[low_pos]):         # Armijo Condition
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if f_new >= float(f_prev + c2 * t * gtd_prev):                                       # Modified Armijo Condition
                # Goldstein conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]

    return float(t)
