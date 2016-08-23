# Serialization
`Merlin` supports serialize / deserialize objects in HDF5 format.

## Save
To save objects,
```julia
h5 = HDF5Dict("name"=>g)
Merlin.save("<path>", h5)
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

```@docs

```
