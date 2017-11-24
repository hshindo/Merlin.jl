# Initializers
Initializers provides the way to set the initial random weights of Merlin functions.

```julia
f = Linear(Float32, 100, 100, init_W=Xavier(), init_b=Fill(0))
```

```@index
Pages = ["initializers.md"]
```

```@docs
Fill
Normal
Orthogonal
Uniform
Xavier
```
