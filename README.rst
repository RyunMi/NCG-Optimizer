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

The implementation of all basic methods is based on the book "Nonlinear Conjugate Gradient Method" [#NCGM]_.

.. [#NCGM] Y.H. Dai and Y. Yuan (2000), Nonlinear Conjugate Gradient Methods, Shanghai Scientific and Technical Publishers, Shanghai. (in Chinese)

Linear Conjugate Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------------------------------------------------------------------------------------------------+
| .. image:: https://raw.githubusercontent.com/RyunMi/NCG-optimizer/master/docs/LCG.png                                       |
+-----------------------------------------------------------------------------------------------------------------------------+