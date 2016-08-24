# Merlin.jl

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).
It aims to provide a fast, flexible and compact deep learning library for machine learning.

See README.md for basic usage.

Basically,

1. Wrap your data with `Var` type.
2. Apply functions to the `Var`.
3. Compute gradient if necessary.

```julia
x = Var(rand(Float32,10,5))
zerograd!(x)
f = Linear(Float32,10,3)
y = f(x)
gradient!(y)
```
