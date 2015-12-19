.. currentmodule:: Mariana

************
Linear Layer
************

.. math::

  y=Wx+b

Functions
---------

.. function:: Linear(weight, bias, gradweight, gradbias)

.. function:: Linear(weight, bias)

.. function:: Linear(::Type{T}, insize, outsize)

  insize::Int: size of input

  outsize::Int: size of output

  initialize weight with Gaussian distribution

.. function:: forward(l::Linear, input::Matrix)

.. function:: forward(l::Linear, input::Array)

Example
---------

.. code-block:: julia

  l = Linear(Float32, 100, 50)
  x = rand(Float32, 100, 3)
  y = forward(l, x)
