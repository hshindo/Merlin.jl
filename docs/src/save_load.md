# Save and Load
`Merlin` supports saving and loading objects in HDF5 format.

## Save
To save objects, use `h5save` function.

```@docs
h5save
```

### 👉 Example
```julia
x = Embeddings(Float32,10000,100)
h5save("<filename>", x)
```

A network structure can be saved as well:
### 👉 Example
```julia
T = Float32
ls = [Linear(T,10,7), Linear(T,7,3)]
g = @graph begin
    x = ls[1](:x)
    x = relu(x)
    x = ls[2](x)
    x
end
h5save("<filename>", g)
```

## Load
To load objects, use `h5load` function.

```@docs
h5load
```

## Saving Custom Objects
It requires to implement `h5convert` and `h5load!` functions.

```@docs
h5dict
```
