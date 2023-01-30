=============
NCG-optimizer
=============

**NCG-optimizer** is a set of optimizer about *nonliear conjugate gradient* in Pytorch.

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

        from ncg_optimizer import LCG
        
        # model = Your Model
        optimizer = LCG(model.parameters(), eps=1e-5)
        def closure():
            optimizer.zero_grad()
            loss_fn(model(input), target).backward()
            return loss_fn
        optimizer.step(closure)

Fletcher-Reeves
^^^^^^^^^^^^^^^


References
==========

.. [#NCGM] Y.H. Dai and Y. Yuan (2000), Nonlinear Conjugate Gradient Methods, Shanghai Scientific and Technical Publishers, Shanghai. (in Chinese)
.. [#NO1] Nocedal J, Wright S J. Line search methods[J]. Numerical optimization, 2006: 30-65.
.. [#NO2] Nocedal J, Wright S J. Conjugate gradient methods[J]. Numerical optimization, 2006: 101-134. 
.. [#CGNO] Pytlak R. Conjugate gradient algorithms in nonconvex optimization[M]. Springer Science & Business Media, 2008.