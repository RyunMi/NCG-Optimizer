#FR
############共轭梯度算法#############
import numpy as np
import matplotlib.pyplot as plt
def Fun(x, y):
    a = 2.0
    b = 1.0
    z = (x ** 2) / a  + (y ** 2) / b + 2 * x * y -x
    return z

def grad_Fun(x, y):
    '''
    求函数的梯度
    '''
    delta_x = 1e-6      #x方向差分小量
    delta_y = 1e-6      #y方向差分小量
    grad_x = (Fun(x + delta_x, y) - Fun(x - delta_x, y)) / (2.0 * delta_x)
    grad_y = (Fun(x, y + delta_y) - Fun(x, y - delta_y)) / (2.0 * delta_y)
    grad_xy = np.array([grad_x, grad_y])
    return grad_xy

def get_StepLength(array_xy, array_d):
    a0 = 1.0          
    e0 = 1e-6          
    delta_a = 1e-6     
    while(1):
        new_a = array_xy + a0*array_d
        new_a_l = array_xy + (a0-delta_a)*array_d
        new_a_h = array_xy + (a0+delta_a)*array_d
        diff_a0 = (Fun(new_a_h[0], new_a_h[1]) - Fun(new_a_l[0], new_a_l[1])) / (2.0 * delta_a)
        if np.abs(diff_a0) < e0:
            break
        ddiff_a0 = (Fun(new_a_h[0], new_a_h[1]) + Fun(new_a_l[0], new_a_l[1]) - 2.0 * Fun(new_a[0], new_a[1])) / (delta_a * delta_a)
        a0 = a0 - diff_a0/ddiff_a0
    return a0

def huatu(new_xy_history):
    x = np.linspace(-40.0, 40.0, 100)
    y = np.linspace(-40.0, 80.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = Fun(X, Y)
    plt.figure(dpi=300)
    plt.xlim(-40.0, 40.0)
    plt.ylim(-40.0, 80.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contour(X, Y, Z, 40)
    plt.plot(new_xy_history[:, 0], new_xy_history[:, 1], marker='.', ms=10)
    plt.show()
    xy_count = new_xy_history.shape[0]
    for i in range(xy_count):
        if i == xy_count-1:
            break
        dx = (new_xy_history[i + 1][0] - new_xy_history[i][0]) * 0.6
        dy = (new_xy_history[i + 1][1] - new_xy_history[i][1]) * 0.6
        plt.arrow(new_xy_history[i][0], new_xy_history[i][1], dx, dy, width=0.8)

def mainFRCG():
    '''
    使用CG算法优化，用FR公式计算组合系数
    '''
    xy_history = []                           
    x1 = np.array([20.0, 60.0])                  
    g1 = grad_Fun(x1[0], x1[1])
    d1 = -1.0 * g1                             
    e0 = 1e-6                                    
    xy = x1
    while(1):
        xy_history.append(xy)
        g1 = grad_Fun(xy[0], xy[1])
        tag = np.abs(g1) < e0
        if tag.all():
            break
        alpha = get_StepLength(xy, d1)
        xy_new = xy + alpha * d1
        gk = grad_Fun(xy_new[0], xy_new[1])
        b = np.dot(gk, gk) / np.dot(g1, g1)      
        d1 = b * d1 - gk
        xy = xy_new
    xy_new_xy_history = np.array(xy_history)
    huatu(xy_new_xy_history)
    return xy_new_xy_history

mainFRCG()
