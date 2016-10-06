# Save and Load
`Merlin` supports saving and loading objects in HDF5 format.
* For saving objects provided by Merlin, use `Merlin.save` and `Merlin.load` functions.
* For other complex objects, it is recommended to use `JLD.save` and `JLD.load` functions provided by [JLD.jl](https://github.com/JuliaIO/JLD.jl).

```@docs
save
load
```

For example,
```julia
x = Embeddings(Float32,10000,100)
Merlin.save("embedding.h5", "w", "x", x)
```

A graph structure can be saved as well:
```julia
T = Float32
x = Var()
y = Linear(T,10,7)(x)
y = relu(y)
y = Linear(T,7,3)(y)
g = Graph(y, x)
Merlin.save("graph.h5", "g", g)
```

The saved HDF5 file is as follows:
<p><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/graph.h5.png"></p>

## Custom Serialization
It requires to implement `h5convert` function for custom serialization/deserialization.
See Merlin sources for details.
