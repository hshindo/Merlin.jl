using HDF5

"""
    save(path::String, name::String, obj, [mode="w"])

Save an object in Merlin HDF5 format.
* mode: "w" (overrite) or "r+" (append)
"""
function save(path::String, data::Pair{String}...; mode="w")
    mkpath(dirname(path))
    h5open(path, mode) do h
        for (k,v) in data
            h5save(h, k, v)
        end
    end
    nothing
end
#save(path::String, name::String, g::Graph; mode="w") = save(path, Dict(name=>g), mode=mode)
#function save(path::String, obj; mode="w")
#    dict = Dict(name=>string(getfield(obj,name)) for name in fieldnames(obj))
#    save(path, dict, mode=mode)
#end

"""
    load(path::String, name::String)

Load an object from Merlin HDF5 format.
"""
function load(path::String, name::String)
    h5open(path, "r") do h
        h5load(h[name])
    end
end

h5convert(x::Void) = string(x)
h5convert(x::Char) = string(x)
h5convert(x::Symbol) = string(x)
h5convert(x::DataType) = string(x)
h5convert(x::Function) = string(x)
h5convert(x::HDF5.BitsKindOrString) = x
function h5convert{T,N}(x::Array{T,N})
    if T <: HDF5.BitsKindOrString
        x
    elseif N == 1
        Dict(i=>x[i] for i=1:length(x))
    else
        throw("x::$(typeof(x)) is not supported.")
    end
end
h5convert(x::Tuple) = Dict(i=>x[i] for i=1:length(x))

h5convert(x) = Dict(name=>getfield(x,name) for name in fieldnames(x))

h5convert(::Type{Void}, x) = nothing
h5convert(::Type{Char}, x) = x[1]
h5convert(::Type{Symbol}, x) = parse(x)
h5convert(::Type{DataType}, x) = eval(parse(x))
h5convert(::Type{Function}, x) = eval(parse(x))
h5convert{T<:HDF5.BitsKindOrString}(::Type{T}, x) = x
function h5convert{T,N}(::Type{Array{T,N}}, x)
    if T <: HDF5.BitsKindOrString
        x
    elseif N == 1
        data = Array(T, length(x))
        for (k,v) in x
            data[parse(Int,k)] = v
        end
        data
    else
        throw("Invalid data: $x")
    end
end
function h5convert{T<:Tuple}(::Type{T}, x)
    data = Array(Any, length(x))
    for (k,v) in x
        data[parse(Int,k)] = v
    end
    tuple(data...)
end

function h5convert2(T, x)
    values = map(name -> x[string(name)], fieldnames(T))
    T(values...)
end

h5type{T<:Function}(::Type{T}) = "Function"
h5type(T) = string(T)

function h5save(group, key::String, obj)
    T = typeof(obj)
    h5obj = h5convert(obj)
    if isa(h5obj, Dict)
        g = g_create(group, key)
        attrs(g)["#JULIA_TYPE"] = h5type(T)
        for (k,v) in h5obj
            h5save(g, string(k), v)
        end
    else
        group[key] = h5obj
        attrs(group[key])["#JULIA_TYPE"] = h5type(T)
    end
end

function h5load(group::HDF5Group)
    dict = Dict()
    for name in names(group)
        dict[name] = h5load(group[name])
    end
    attr = read(attrs(group), "#JULIA_TYPE")
    T = eval(current_module(), parse(attr))
    h5convert(T, dict)
end

function h5load(dataset::HDF5Dataset)
    data = read(dataset)
    if exists(attrs(dataset), "#JULIA_TYPE")
        attr = read(attrs(dataset), "#JULIA_TYPE")
        T = eval(current_module(), parse(attr))
        h5convert(T, data)
    else
        data
    end
end
