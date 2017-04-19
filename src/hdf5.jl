using HDF5
import HDF5.BitsKindOrString

"""
    save(path::String, name::String, obj, [mode="w"])

Save an object in Merlin HDF5 format.
* mode: "w" (overrite) or "r+" (append)
"""
function save(path::String, data::Pair{String}...; mode="w")
    info("Saving objects...")
    mkpath(dirname(path))
    h5open(path, mode) do h
        for (k,v) in data
            h5save(h, k, v)
        end
    end
    nothing
end

"""
    load(path::String, name::String)

Load an object from Merlin HDF5 format.
"""
function load(path::String, name::String)
    h5open(path, "r") do h
        h5load(h[name])
    end
end

readas(::Type{Void}, x) = nothing
writeas(x::Void) = string(x)

readas(::Type{Char}, x) = x[1]
writeas(x::Char) = string(x)

readas(::Type{Symbol}, x) = parse(x)
writeas(x::Symbol) = string(x)

readas(::Type{DataType}, x) = eval(parse(x))
writeas(x::DataType) = string(x)

readas(::Type{Function}, x) = eval(parse(x))
writeas(x::Function) = string(x)

readas{T<:BitsKindOrString}(::Type{T}, x) = x
writeas(x::BitsKindOrString) = x

function readas{T,N}(::Type{Array{T,N}}, x)
    if T <: BitsKindOrString
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
function writeas{T,N}(x::Array{T,N})
    if T <: BitsKindOrString
        x
    elseif N == 1
        Dict(i=>x[i] for i=1:length(x))
    else
        throw("x::$(typeof(x)) is not supported.")
    end
end

function readas{K<:BitsKindOrString,V<:BitsKindOrString}(::Type{Dict{K,V}}, x)
    dict = Dict{K,V}()
    for line in split(x, "\n")
        items = split(chomp(line), "\t")
        k = K <: String ? String(items[1]) : parse(K,items[1])
        v = V <: String ? String(items[2]) : parse(V,items[2])
        dict[k] = v
    end
    dict
end
function writeas{K<:BitsKindOrString,V<:BitsKindOrString}(x::Dict{K,V})
    lines = String[]
    for (k,v) in x
        push!(lines, "$k\t$v")
    end
    join(lines, "\n")
end

function readas{T<:Tuple}(::Type{T}, x)
    data = Array(Any, length(x))
    for (k,v) in x
        data[parse(Int,k)] = v
    end
    tuple(data...)
end
writeas(x::Tuple) = Dict(i=>x[i] for i=1:length(x))

function readas(T::Type, x)
    values = map(name -> x[string(name)], fieldnames(T))
    T(values...)
end
writeas(x) = Dict(name=>getfield(x,name) for name in fieldnames(x))

h5type{T<:Function}(::Type{T}) = "Function"
h5type(T) = string(T)

function h5save(group, key::String, obj)
    T = typeof(obj)
    h5obj = writeas(obj)
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
    readas(T, dict)
end

function h5load(dataset::HDF5Dataset)
    data = read(dataset)
    if exists(attrs(dataset), "#JULIA_TYPE")
        attr = read(attrs(dataset), "#JULIA_TYPE")
        T = eval(current_module(), parse(attr))
        readas(T, data)
    else
        data
    end
end
