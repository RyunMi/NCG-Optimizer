# LCG
############共轭梯度算法#############
import numpy as np
import matplotlib.pyplot as plt
def Fun(x, y):
    a = 2/3
    b = 2
    z = (x ** 2) / a + (y ** 2) / b - x * y -2*x
    return z

def plotResult(array_xy_history):
    x = np.linspace(-4.0, 4.0, 100)
    y = np.linspace(-4.0, 8.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = Fun(X, Y)
    plt.figure(dpi=300)
    plt.xlim(-4.0, 4.0)
    plt.ylim(-4.0, 8.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contour(X, Y, Z, 40)
    plt.plot(array_xy_history[:,0], array_xy_history[:,1], marker='.', ms=10)
    xy_count = array_xy_history.shape[0]
    for i in range(xy_count):
        if i == xy_count-1:
            break
        dx = (array_xy_history[i+1][0] - array_xy_history[i][0])*0.6
        dy = (array_xy_history[i+1][1] - array_xy_history[i][1])*0.6
        plt.arrow(array_xy_history[i][0], array_xy_history[i][1], dx, dy, width=0.8)

def mainCG():
    xy_history = []
    A = np.array([[3,-1],
    [-1,1]])
    b=[-2,0]
    x1 = np.array([2.0, 1.0])  
    r1 = A*x1-b                        
    p1 = -1.0*r1                            
    e0 = 1e-6                                    
    while(1):
        xy = x1
        xy_history.append(xy)
        rk1 = r1
        pk1 = p1
        alpha = -np.dot(rk1,rk1)/np.dot(pk1, np.dot(A, pk1))
        x2 = x1 + alpha*pk1
        rk2 = A*x2 -b
        tag_reach = np.abs(rk2) < e0
        if tag_reach.all():
            break
        b = np.dot(rk2, rk2)/np.dot(rk1, rk1)       
        pk2 = b*pk1 - rk2
        xy = x2
    new_xy_history = np.array(xy_history)
    plotResult(new_xy_history)
    return new_xy_history

mainCG()
