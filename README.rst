=============
NCG-optimizer
=============

**NCG-optimizer** is a set of optimizer about *nonliear conjugate gradient* in Pytorch.

Install
=======

::

    $ pip install ncg-optimizer

Example
=======

.. code-block:: python

    from ncg-optimizer import PRP

    # model = Your Model
    optimizer = PRP(model.parameters())

Supported Optimizers
====================

Basic Methods
-------------

The implementation of all basic methods is based on the book Nonlinear Conjugate Gradient Method[#NCGM]_ .

.. [#NCGM] Dai Y H, Yuan Y. Nonlinear conjugate gradient methods[J]. Shanghai Science and Technology Publisher, Shanghai, 2000. 