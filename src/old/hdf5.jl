using HDF5

import Base.convert

doc"""
    H5Object

* typename
* data
"""
struct H5Object
    typename
    data
end

convert(::Type{H5Object}, x::Union{Real,String}) = H5Object(typeof(x), x)
convert(::Type{H5Object}, x::Union{Void,Char,Symbol,DataType,Bool}) = H5Object(typeof(x), string(x))
convert(::Type{H5Object}, x::Function) = H5Object(Function, string(x))

function convert(::Type{H5Object}, x::Array{T,N}) where {T,N}
    if T <: Union{Real,String}
        H5Object(typeof(x), x)
    elseif N == 1
        H5Object(typeof(x), Dict(string(i)=>x[i] for i=1:length(x)))
    else
        throw("Not supported yet.")
    end
end

function convert(::Type{H5Object}, x::Tuple)
    H5Object(Tuple, Dict(string(i)=>x[i] for i=1:length(x)))
end
convert(::Type{H5Object}, x::Dict) = H5Object(typeof(x), x)
function convert(::Type{H5Object}, x::T) where T
    m = Base.datatype_module(T)
    if m == Core || m == Base
        throw("Type $T is not serializable.")
    else
        d = Dict(string(name)=>getfield(x,name) for name in fieldnames(x))
        H5Object(T, d)
    end
end

convert(::Type{<:Union{Real,String}}, o::H5Object) = o.data
convert(::Type{T}, o::H5Object) where T<:Union{Void,DataType,Function} = eval(parse(o.data))
convert(::Type{Char}, o::H5Object) = Vector{Char}(o.data)[1]
convert(::Type{Bool}, o::H5Object) = parse(Bool, o.data)
convert(::Type{Symbol}, o::H5Object) = parse(o.data)
convert(::Type{Function}, o::H5Object) = eval(parse(o.data))

function convert(::Type{Array{T,N}}, o::H5Object) where {T,N}
    if T <: Union{Real,String}
        o.data
    elseif N == 1
        dict = o.data
        data = Array{T}(length(dict))
        for (k,v) in dict
            data[parse(Int,k)] = v
        end
        data
    else
        throw("Not supported yet.")
    end
end

function convert(::Type{T}, o::H5Object) where T<:Tuple
    dict = o.data
    data = Array{Any}(length(dict))
    for (k,v) in dict
        data[parse(Int,k)] = v
    end
    tuple(data...)
end

convert(::Type{Dict{K,V}}, o::H5Object) where {K,V} = Dict{K,V}(o.data)

function convert(::Type{T}, o::H5Object) where T
    m = Base.datatype_module(T)
    if m == Core || m == Base
        throw("Type $T is not deserializable.")
    else
        dict = o.data
        values = map(name -> dict[string(name)], fieldnames(T))
        T(values...)
    end
end

"""
    save(path::String, obj, [mode="w"])

Save an object in Merlin HDF5 format.
* mode: "w" (overrite) or "r+" (append)
"""
function save(path::String, obj, mode="w")
    info("Saving $path...")
    h5obj = convert(H5Object, obj)
    h5open(path,mode) do h
        h["JULIA_VERSION"] = string(Base.VERSION)
        h["MERLIN_VERSION"] = string(Merlin.VERSION)
        if isa(h5obj.data, Dict)
            g = g_create(h, "OBJECT")
            h5save(g, h5obj)
        else
            h["OBJECT"] = h5obj.data
            attrs(h)["JULIA_TYPE"] = string(h5obj.typename)
        end
    end
    nothing
end

function h5save(group::HDF5Group, h5obj::H5Object)
    attrs(group)["JULIA_TYPE"] = string(h5obj.typename)
    for (k,v) in h5obj.data
        h5v = convert(H5Object, v)
        if isa(h5v.data, Dict)
            h5save(g_create(group,k), h5v)
        else
            group[k] = h5v.data
            attrs(group[k])["JULIA_TYPE"] = string(h5v.typename)
        end
    end
end

"""
    load(path::String, name::String)

Load an object from Merlin HDF5 format.
"""
function load(path::String)
    info("Loading $path...")
    dict = h5open(path, "r") do h
        d = Dict{String,Any}()
        for name in names(h)
            d[name] = h5load(h[name])
        end
        d
    end
    dict["OBJECT"]
end

function h5load(group::HDF5Group)
    dict = Dict()
    for name in names(group)
        dict[name] = h5load(group[name])
    end
    attr = read(attrs(group), "JULIA_TYPE")
    T = eval(current_module(), parse(attr))
    convert(T, H5Object(T,dict))
end

function h5load(dataset::HDF5Dataset)
    data = read(dataset)
    if exists(attrs(dataset), "JULIA_TYPE")
        attr = read(attrs(dataset), "JULIA_TYPE")
        T = eval(current_module(), parse(attr))
        convert(T, H5Object(T,data))
    else
        data
    end
end
