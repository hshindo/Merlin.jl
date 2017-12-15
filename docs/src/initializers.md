# Initializers
```@index
Pages = ["initializers.md"]
```

Initializers provides a way to set the initial weights of Merlin functions.

```julia
f = Linear(Float32, 100, 100, init_W=Xavier(), init_b=Fill(0))
```

```@autodocs
Modules = [Merlin]
Pages   = ["initializer.jl"]
```
