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

    # model = You Model
    optimizer = PRP(model.parameters())

Supported Optimizers
====================