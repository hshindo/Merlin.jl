# Optimizers

```julia
x = zerograd(rand(Float32,5,4))
opt = SGD(0.001)
opt(x)
```

```@index
Pages = ["optimizers.md"]
```

```@autodocs
Modules = [Merlin]
Pages = ["optimizers.jl"]
```
