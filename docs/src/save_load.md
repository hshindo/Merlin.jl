# Save and Load
`Merlin` supports saving and loading objects in HDF5 format.

## Save
To save objects, use `h5save` function.

```julia
x = Embeddings(Float32,10000,100)
h5save("<filename>", x)
```

A network structure can be saved as well:
```julia
T = Float32
ls = [Linear(T,10,7), Linear(T,7,3)]
g = @graph begin
    x = ls[1](:x)
    x = relu(x)
    x = ls[2](x)
    x
end
h5save("<filename>", x)
```

```@docs
h5save
```

## Load
To load objects, use `h5load` function.
```julia
x = h5load("<filename>")
```

```@docs
h5load
```

## Custom
It requires `h5convert` and `h5load!` functions.

```@docs
h5dict
```
