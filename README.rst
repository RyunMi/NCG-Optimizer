=============
NCG-optimizer
=============

**NCG-optimizer** is a set of optimizer about *nonlinear conjugate gradient* in Pytorch.

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

The linear conjugate gradient(LCG) method is only applicable to linear equation solving problems. It converts linear equations into quadratic functions, so that the problem can be solved iteratively without inverting the coefficient matrix.

.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/LCG.png
        :width: 800px

.. code-block:: python

        import ncg_optimizer as optim
        
        # model = Your Model
        
        optimizer = optim.LCG(model.parameters(), eps=1e-5)
        def closure():
            optimizer.zero_grad()
            loss_fn(model(input), target).backward()
            return loss_fn
        optimizer.step(closure)

Nonlinear Conjugate Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/NCG.png
        :width: 800px

Fletcher-Reeves Method
""""""""""""""""""""""
The Fletcher-Reeves conjugate gradient method ( **FR Method** ) is the earliest nonlinear conjugate gradient method. 
It was obtained by Fletcher and Reeves in 1964 by extending the conjugate gradient method for solving linear equations to solve optimization problems. 

The scalar parameter update formula of the FR method is as follows:

$$ \\beta_k^{F R}=\\frac{g_{k+1}^T g_{k+1}}{g_k^T g_k}$$

The convergence analysis of FR method is often closely related to its selected line search. 
The FR method of exact line search is used to converge the general nonconvex function. 
The FR method of strong Wolfe inexact line search method (c2 <= 0.5) is adopted to globally converge to the general nonconvex function. 
The generalized Wolfe or Armijo inexact line search FR method is globally convergent for general nonconvex functions.

.. code-block:: python

        
        import ncg_optimizer as optim

        # model = Your Model

        optimizer = optim.FR(
            model.parameters(), line_search = 'Wolfe', 
            c1 = 1e-4, c2 = 0.5, lr = 0.5, eta = 5,)
        def closure():
            optimizer.zero_grad()
            loss_fn(model(input), target).backward()
            return loss_fn
        optimizer.step(closure)

References
==========

.. [#NCGM] Y.H. Dai and Y. Yuan (2000), Nonlinear Conjugate Gradient Methods, Shanghai Scientific and Technical Publishers, Shanghai. (in Chinese)
.. [#NO1] Nocedal J, Wright S J. Line search methods[J]. Numerical optimization, 2006: 30-65.
.. [#NO2] Nocedal J, Wright S J. Conjugate gradient methods[J]. Numerical optimization, 2006: 101-134. 
.. [#CGNO] Pytlak R. Conjugate gradient algorithms in nonconvex optimization[M]. Springer Science & Business Media, 2008.
