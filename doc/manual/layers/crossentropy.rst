.. currentmodule:: Mariana

*******************
CrossEntropy Layer
*******************

Compute the cross-entropy between a true distribution math:`p` and
the specified distribution math:`q`:

.. math::

  \mathrm{H}(p,q)=-\sum_{x}p(x)\log q(x)

Functions
---------

  .. function:: CrossEntropy(weight::Vector{Int})

    Create CrossEntropy layer.

  .. function:: forward(l::CrossEntropy, input::Matrix{T})
