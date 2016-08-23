# Save and Load
`Merlin` supports saving and loading objects (network structures, parameters, etc.) in HDF5 format.

## Save
```julia
x =
Merlin.save("<filename>", x)
```

## Load
To deserialize objects,
```julia
Merlin.load()
```

## Index
```@index
Pages = ["functions.md"]
```

## Activation Functions
```@docs
relu
sigmoid
tanh
```

## How to serialize your object?
It requires
* HDF5Dict(x)
* load_hdf5(::Type{T}, x::Dict)

See examples.

```@docs

```
