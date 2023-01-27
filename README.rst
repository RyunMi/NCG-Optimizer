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

The implementation of all basic methods is based on the book "Nonlinear Conjugate Gradient Method" [#NCGM]_.

.. [#NCGM] Y.H. Dai and Y. Yuan (2000), Nonlinear Conjugate Gradient Methods, Shanghai Scientific and Technical Publishers, Shanghai. (in Chinese)

Linear Conjugate Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^

The linear conjugate gradient(LCG) method is only applicable to linear equation solving problems. 
It converts linear equations into quadratic functions, 
so that the problem can be solved iteratively without inverting the coefficient matrix.

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