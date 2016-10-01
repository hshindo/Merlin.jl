to_hdf5(x::Vector{Any}) = Dict(i => x[i] for i=1:length(x))
function from_hdf5(::Type{Vector{Any}}, obj)
    x = Array(Any, length(obj))
    for (k,v) in obj
        x[parse(Int,k)] = v
    end
    x
end

to_hdf5(x::Dict) = x
from_hdf5{T<:Dict}(::Type{T}, x) = x

to_hdf5(x::Symbol) = string(x)
from_hdf5(::Type{Symbol}, x) = Symbol(x)

to_hdf5(x::Function) = string(typeof(x).name.mt.name)
from_hdf5{T<:Function}(::Type{T}, x) = eval(parse(x))

to_hdf5(x::DataType) = string(x)
from_hdf5(::Type{DataType}, x::String) = eval(parse(x))

to_hdf5(x::Void) = string(x)
from_hdf5(::Type{Void}, x) = nothing

function write_hdf5{T}(group, key::String, obj::T)
    if T <: HDF5.BitsKindOrString ||
        (T <: Array && eltype(obj) <: HDF5.BitsKindOrString)
        group[key] = obj
    else
        h5obj = to_hdf5(obj)
        if typeof(h5obj) <: Dict
            g = g_create(group, key)
            attrs(g)["JULIA_TYPE"] = string(T)
            for (k,v) in h5obj
                write_hdf5(g, string(k), v)
            end
        else
            group[key] = h5obj
            t =
            attrs(group[key])["JULIA_TYPE"] = string(T)
        end
    end
end

function load(filename::String, key::String)
    h5open(filename, "r") do h
        read_hdf5(h[key])
    end
end

function read_hdf5(group::HDF5Group)
    dict = Dict()
    for g in group
        dict[name(g)] = read_hdf5(g)
    end
    T = from_hdf5(DataType, attrs(group)["JULIA_TYPE"])
    from_hdf5(T, dict)
end

function read_hdf5(dataset::HDF5Dataset)
    data = read(dataset)
    if exists(attrs(dataset), "JULIA_TYPE")
        attr = read(attrs(dataset),"JULIA_TYPE")
        println(attr)
        T = from_hdf5(DataType, attr)
        from_hdf5(T, data)
    else
        data
    end
end

function save(filename::String, key::String, obj)
    h5open(filename, "w") do h
        write_hdf5(h, key, obj)
    end
    nothing
end
