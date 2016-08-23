# Save and Load
`Merlin` supports saving and loading objects (network structures, parameters, etc.) in HDF5 format.

## Save
```julia
Merlin.save("<filename>", x)
```

## Load
To deserialize objects,
```julia
Merlin.load()
```

## How to serialize your object?
It requires
* HDF5Dict(x)
* load_hdf5(::Type{T}, x::Dict)

See examples.
