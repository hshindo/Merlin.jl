.. currentmodule:: Mariana

**************
Concat Layer
**************

Concatenate input arrays along the specified dimension.

Functions
---------

.. function:: Concat(dim::Int)

  Create Concat layer.

.. function:: forward(l::Concat, inputs::Vector)

Example
---------

.. code-block:: julia

  l = Concat(1)
  a = rand(Float32, 10, 2)
  b = rand(Float32, 10, 2)
  inputs = []
  push!(inputs, a)
  push!(inputs, b)
  forward(l, inputs)
