.. currentmodule:: Mariana

****************
LogSoftmax Layer
****************

.. math::

  y=\frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})},\;i=1,\ldots,n

Functions
---------

.. function:: LogSoftmax(weight, bias, gradweight, gradbias)

.. function:: forward(l::LogSoftmax, input::Array)
