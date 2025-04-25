=============
NCG-Optimizer
=============
.. image:: https://github.com/RyunMi/NCG-optimizer/actions/workflows/publish.yml/badge.svg
    :target: https://github.com/RyunMi/NCG-optimizer/actions?query=workflow
.. image:: https://img.shields.io/pypi/pyversions/ncg-optimizer.svg
    :target: https://pypi.org/project/ncg-optimizer
.. image:: https://img.shields.io/pypi/v/ncg-optimizer.svg
    :target: https://pypi.python.org/pypi/ncg-optimizer
.. image:: https://static.pepy.tech/badge/ncg-optimizer
    :target: https://pepy.tech/project/ncg-optimizer
.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0

**NCG-Optimizer** is a set of optimizer about *nonlinear conjugate gradient* in PyTorch.

Inspired by `@jettify <https://github.com/jettify/pytorch-optimizer>`__ and `@kozistr <https://github.com/kozistr/pytorch_optimizer>`__.

Install
=======

::

    $ pip install ncg_optimizer

Supported Optimizers
====================

Basic Methods
-------------

The theoretical analysis and implementation of all basic methods is based on the "Nonlinear Conjugate Gradient Method" [#NCGM]_ , "Numerical Optimization" ([#NO1]_ [#NO2]_) and "Conjugate gradient algorithms in nonconvex optimization"[#CGNO]_.

Linear Conjugate Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^

The Linear Conjugate Gradient(**LCG**) method is only applicable to linear equation solving problems. It converts linear equations into quadratic functions, so that the problem can be solved iteratively without inverting the coefficient matrix.

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/LCG.png
        :width: 800px

.. code-block:: python

        import ncg_optimizer as optim
        
        # model = Your Model
        
        optimizer = optim.LCG(model.parameters(), eps=1e-5)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)

Nonlinear Conjugate Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/NCG.png
        :width: 800px

Fletcher-Reeves Method
""""""""""""""""""""""
The Fletcher-Reeves conjugate gradient(**FR**) method  is the earliest nonlinear conjugate gradient method. 
It was obtained by Fletcher and Reeves in 1964 by extending the conjugate gradient method for solving linear equations to solve optimization problems. 

The scalar parameter update formula of the FR method is as follows:

$$ \\beta_k^{F R}=\\frac{g_{k+1}^T g_{k+1}}{g_k^T g_k}$$

The convergence analysis of FR method is often closely related to its selected line search. 
The FR method of exact line search is used to converge the general nonconvex function. 
The FR method of strong Wolfe inexact line search method $c_2 \\leq 0.5$ is adopted to globally converge to the general nonconvex function. 
The generalized Wolfe or Armijo inexact line search FR method is globally convergent for general nonconvex functions.

.. code-block:: python

        
        optimizer = optim.BASIC(
            model.parameters(), method = 'FR',
            line_search = 'Strong_Wolfe', c1 = 1e-4, 
            c2 = 0.5, lr = 0.2, max_ls = 25)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)

Polak-Ribiere-Polyak Method
"""""""""""""""""""""""""""

The  Polak-Ribiere-Polyak(**PRP**) method is a nonlinear conjugate gradient method proposed independently by Polak, Ribiere and Polyak in 1969. 
The PRP method is one of the conjugate gradient methods with the best numerical performance. 
When the algorithm produces a small step, the search direction $d_k$ defined by the PRP method automatically approaches the negative gradient direction, 
thus effectively avoiding the disadvantage that the FR method may continuously produce small steps.

The scalar parameter update formula of the PRP method is as follows:

$$ \\beta_k^{PRP}=\\frac{g_{k}^{T}(g_{k}-g_{k-1})}{\\lVert g_{k-1}\\rVert^2}$$

The convergence analysis of the PRP method is often closely related to the selected line search. When the step size $s_k = x_{k+1} - x_{k} \\to 0$ is regarded as a measure of global convergence, 
the PRP method of exact line search is used to converge the uniformly convex function under this benchmark. 
The PRP method using Armijo-type inexact line search method converges globally for general nonconvex functions. 
The PRP $^+$ method using the strong Wolfe($0 < c_2 < \\frac{1}{4}$) inexact line search method converges globally for general nonconvex functions. 
The PRP method with some constant step size factor (involving Lipschitz constant) inexact line search method converges globally for general nonconvex functions.

.. code-block:: python


        optimizer = optim.BASIC(
            model.parameters(), method = 'PRP',
            line_search = 'Armijo', c1 = 1e-4, 
            c2 = 0.9, lr = 1, rho = 0.5,)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)

Hestenes-Stiefel Method
"""""""""""""""""""""""

Another famous conjugate gradient method Hestenes-Stiefel(**HS**) method was proposed by Hestenes and Stiefel.
The scalar parameter update formula of the HS method is as follows:

$$ \\beta_{k}^{HS}=\\frac{g_{k}^{T}(g_{k}-g_{k-1})}{(g_{k}-g_{k-1})^Td_{k-1}} $$

Compared with the PRP method, an important property of the HS method is that the conjugate relation 
$d_k^T(g_{k}-g_{k-1}) = 0$ always holds regardless of the exact of the line search. 
However, the theoretical properties and computational performance of the HS method are similar to those of the PRP method.

The convergence analysis of the HS method is often closely related to the selected line search. 
If the $f(x)$ level set is bounded, its derivative is Lipschitz continuous and satisfies the sufficient descent condition, 
then the HS method with Wolfe inexact line search method is globally convergent. 
The HS $^+$ method with the strong Wolfe ($0 < c_2 < \\frac{1}{3}$) inexact line search method converges globally for general nonconvex functions.

.. code-block:: python


        optimizer = optim.BASIC(
            model.parameters(), method = 'HS',
            line_search = 'Strong_Wolfe', c1 = 1e-4, 
            c2 = 0.4, lr = 0.2, max_ls = 25,)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)

Conjugate Descent Method
""""""""""""""""""""""""
Conjugate Descent (**CD**) was first introduced by Fletcherl in 1987. 
It can avoid the phenomenon that a rising search direction may occur in each iteration 
such as the PRP method and the FR method under certain conditions.

The scalar parameter update formula of the CD method is as follows:

$$ \\beta_{k}^{CD}=\\frac{g_{k}^T g_{k}}{-(g_{k-1})^T d_{k-1}} $$

The convergence analysis of the CD method is often closely related to the selected line search. 
The CD method using the strong Wolfe ($c_2 < 1$) inexact line search method converges globally for general nonconvex functions, 
but the convergence accuracy cannot be guaranteed. 
The CD method using Armijo inexact line search method converges globally for general nonconvex functions.

.. code-block:: python


        optimizer = optim.BASIC(
            model.parameters(), method = 'CD',
            line_search = 'Armijo', c1 = 1e-4, 
            c2 = 0.9, lr = 1, rho = 0.5,)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)

Liu-Storey Method
"""""""""""""""""
Liu-Storey (**LS**) conjugate gradient method is a nonlinear conjugate gradient method 
proposed by Liu and Storey in 1991, which has good numerical performance.

The scalar parameter update formula of the LS method is as follows:

$$ \\beta_{k}^{LS}=\\frac{g_{k}^T (g_{k} - g_{k-1})}{ - g_{k-1}^T d_{k-1}} $$

The convergence analysis of the LS method is often closely related to the selected line search. 
The LS method with strong Wolfe inexact line search method has global convergence property (under Lipschitz condition). 
The LS method using Armijo-type inexact line search method converges globally for general nonconvex functions.

.. code-block:: python


        optimizer = optim.BASIC(
            model.parameters(), method = 'LS',
            line_search = 'Armijo', c1 = 1e-4, 
            c2 = 0.9, lr = 1, rho = 0.5,)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)



Dai-Yuan Method
"""""""""""""""

The Dai-Yuan method (**DY**) was first proposed by Yuhong Dai and Yaxiang Yuan in 1995, which always produces a descent search direction under weaker line search conditions and is globally convergent. 
In addition, good convergence results can be obtained without using strong Wolfe inexact line search but only using Wolfe inexact line search.

The scalar parameter update formula of the DY method is as follows:

$$ \\beta_{k}^{DY}=\\frac{g_{k}^T g_{k}}{(g_{k} - g_{k-1})^T d_{k-1}} $$

The convergence analysis of the DY method is often closely related to the selected line search. 
The DY method using the strong Wolfe inexact line search method can guarantee sufficient descent and global convergence for general nonconvex functions. 
The DY method using the Wolfe inexact line search method converges globally for general nonconvex functions.

.. code-block:: python


        optimizer = optim.BASIC(
            model.parameters(), method = 'DY',
            line_search = 'Strong_Wolfe', c1 = 1e-4, 
            c2 = 0.9, lr = 0.2, max_ls = 25,)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)

Hager-Zhang Method [#HZ]_
"""""""""""""""""""""""""
The Hager-Zhang (**HZ**) method is a new nonlinear conjugate gradient method proposed by Hager and Zhang in 2005. 
It satisfies the sufficient descent condition and has global convergence for strongly convex functions, 
and the search direction approaches the direction of the memoryless BFGS quasi-Newton method.

The scalar parameter update formula of the HZ method is as follows:

$$
\\beta_k^{HZ}=\\frac{1}{d_{k-1}^T (g_{k} - g_{k-1})}((g_{k} - g_{k-1})-2 d_{k-1} \\frac{\\|(g_{k} - g_{k-1}) \\|^2}{d_{k-1}^T (g_{k} - g_{k-1})})^T{g}_{k}
$$

The convergence analysis of the HZ method is often closely related to the selected line search. 
The HZ method with (strong) Wolfe inexact line search method converges globally for general nonconvex functions. 
The HZ $^+$ method using Armijo inexact line search method converges globally for general nonconvex functions.

.. code-block:: python


        optimizer = optim.BASIC(
            model.parameters(), method = 'HZ',
            line_search = 'Strong_Wolfe', c1 = 1e-4, 
            c2 = 0.9, lr = 0.2, max_ls = 25,)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)


Hybrid HS-DY Method
"""""""""""""""""""
Dai and Yuan studied the **HS-DY** hybrid conjugate gradient method of. 
Compared with other hybrid conjugate gradient methods (such as FR + PRP hybrid conjugate gradient method), 
the advantage of this hybrid method is that it does not require the line search to satisfy the strong Wolfe condition, but only the Wolfe condition. 
Their numerical experiments show that the HS-DY hybrid conjugate gradient method performs very well on difficult problems.

The scalar parameter update formula of the HS-DY method is as follows:

$$
\\beta_k^{HS-DY}=\\max (0, \\min (\\beta_k^{HS}, \\beta_k^{DY})))
$$

Regarding the convergence analysis of the HS-DY method, 
the HS-DY method using the Wolfe inexact line search method is globally convergent for general non-convex functions, 
and the performance effect is also better than the PRP method.

.. code-block:: python


        optimizer = optim.BASIC(
            model.parameters(), method = 'HS-DY',
            line_search = 'Armijo', c1 = 1e-4, 
            c2 = 0.9 lr = 1, rho = 0.5,)
        def closure():
            optimizer.zero_grad()
            loss(model(input), target).backward()
            return loss
        optimizer.step(closure)

Line Search
^^^^^^^^^^^
Armijo Line Search [#NO1]_
"""""""""""""""""""""""""""
In order to satisfy the condition that the decrease of the function is at least proportional to the decrease of the tangent, there are:

$$
f\\left(x_k + a_k d_k\\right) \\leqslant f\\left(x_k\\right) + c_1 a_k g_k^T d_k
$$

Among them, $c_1\\in (0,1)$ is generally taken as $c_1 = 10^{-4}$.

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/ArmijoLS.png
        :width: 800px

Curvature Line Search [#NO1]_
""""""""""""""""""""""""""""""
The Armijo condition does not ensure significant progress as it is satisfied for all sufficiently small values of $$α_k$$. So, it often gets stuck far away from the optimal solution. As a result, the Curvature condition requires the current improvement to be greater than $$c_2$$ times the previous one.

$$
\\nabla f\\left(x_k + a_k d_k\\right)^T d_k \\geqslant c_2 \\nabla f_k^T d_k  
$$

where $c_2\\in (c_1,1)$.

Typical values of $$c_2$$ are $$0.9$$ for Newton or quasi-Newton methods and $$0.1$$ for nonlinear conjugate gradient methods.

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/Curvature.png
        :width: 400px

Weak Wolfe Line Search [#NO1]_
"""""""""""""""""""""""""""""""""
Weak Wolfe condition utilizes Armijo as a lower bound solver and Curvature as an upper bound solver.

$$
f\\left(x_k + a_k d_k\\right) \\leqslant f\\left(x_k\\right) + c_1 a_k g_k^T d_k
$$

$$
\\nabla f\\left(x_k + a_k d_k\\right)^T d_k \\geqslant c_2 \\nabla f_k^T d_k  
$$

where $$0 < c_1 < c_2 < 1$$.

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/Weak_Wolfe.png
        :width: 500px
        
Strong Wolfe Line Search [#NO1]_ [#MF]_
"""""""""""""""""""""""""""""""""""""""
Despite Weak Wolfe conditions being considered a reliable method, it often needs many parameter updates to reach the optimal solution. For this reason, Strong Wolfe conditions can speed up the searching process by forcing $$α_k$$ to lie in at least a broad neighborhood of the optimal point.

$$
f\\left(x_k + a_k d_k\\right) \\leqslant f\\left(x_k\\right) + c_1 a_k g_k^T d_k
$$

$$
\|\\nabla f\\left(x_k + a_k d_k\\right)^T d_k\| \\leqslant c_2 \|\\nabla f_k^T d_k\|
$$

where $$0 < c_1 < c_2 < 1$$.

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/Strong_Wolfe.png
        :width: 800px

Goldstein Line Search [#NO1]_ [#MF]_
"""""""""""""""""""""""""""""""""""""""
Goldstein conditions are built on top of Wolfe conditions. It is a great alternative method as it ensures both a sufficient decrement of the objective value and avoids too large parameter updates.

$$
f\\left(x_k + a_k d_k\\right) \\leqslant f\\left(x_k\\right) + c a_k g_k^T d_k
$$

$$
f\\left(x_k + a_k d_k\\right) \\geqslant f\\left(x_k\\right) + (1-c) a_k g_k^T d_k
$$

where $$0 < c < 0.5$$.

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/Goldstein.png
        :width: 600px

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/Zoom.png
        :width: 800px

Contribution
============

Maintained by `@RyunMi <https://github.com/RyunMi>`__, `@Apostolis-Greece <https://github.com/Apostolis-Greece>`__.

References
==========

.. [#NCGM] Y.H. Dai and Y. Yuan (2000), Nonlinear Conjugate Gradient Methods, Shanghai Scientific and Technical Publishers, Shanghai. (in Chinese)
.. [#NO1] Nocedal J, Wright S J. Line search methods[J]. Numerical optimization, 2006: 30-65.
.. [#NO2] Nocedal J, Wright S J. Conjugate gradient methods[J]. Numerical optimization, 2006: 101-134. 
.. [#CGNO] Pytlak R. Conjugate gradient algorithms in nonconvex optimization[M]. Springer Science & Business Media, 2008.
.. [#HZ] Hager W W, Zhang H. A new conjugate gradient method with guaranteed descent and an efficient line search[J]. SIAM Journal on optimization, 2005, 16(1): 170-192.
.. [#MF] Schmidt M. minFunc: unconstrained differentiable multivariate optimization in Matlab[J]. Software available at https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005.
