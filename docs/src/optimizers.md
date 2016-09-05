# Optimizers

A Optimizer provides functions for updating parameters.

For example,
```julia
x1 = Var(rand(Float32,5,4))
x1.grad = rand(Float32,5,4)
opt = SGD(0.001)
opt(x1.data, x1.grad)
```

```@docs
AdaGrad
Adam
SGD
```
