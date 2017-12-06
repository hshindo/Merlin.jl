# Optimizers
Optimizers provides a way to update the weights of `Merlin.Var`.

```julia
x = zerograd(rand(Float32,5,4))
opt = SGD(0.001)
opt(x)
println(x.grad)
```

```@index
Pages = ["optimizers.md"]
```

```@autodocs
Modules = [Merlin]
Pages   = ["optimizer.jl"]
```
