# Optimizers

```julia
x = zerograd(rand(Float32,5,4))
opt = SGD(0.001)
opt(x.data, x.grad)
```

```@index
Pages = ["optimizers.md"]
```

```@docs
Adagrad
Adam
SGD
```
