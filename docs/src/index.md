# Merlin.jl

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).
It aims to provide a fast, flexible and compact deep learning library for machine learning.

See README.md for basic usage.

Basically,

1. Wrap your data with `Var` type.
2. Apply functions to the `Var`.

```julia
x = Var(rand(Float32,10,5))
f = Linear(Float32,10,3)
y = f(x)
```
