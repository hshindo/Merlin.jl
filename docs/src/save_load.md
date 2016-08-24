# Save and Load
`Merlin` supports saving and loading objects in HDF5 format.

## Save
```@docs
h5save
```

For example,
```julia
x = Embeddings(Float32,10000,100)
h5save("<filename>.h5", x)
```

A graph structure can be saved as well:
```julia
T = Float32
ls = [Linear(T,10,7), Linear(T,7,3)]
g = @graph begin
    x = ls[1](:x)
    x = relu(x)
    x = ls[2](x)
    x
end
h5save("<filename>.h5", g)
```

The saved HDF5 file is as follows:
<p><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/sample.h5.png"></p>

## Load
```@docs
h5load
```

## Saving Your Own Objects
It requires to implement `h5convert` and `h5load!` functions.

```@docs
h5dict
```
