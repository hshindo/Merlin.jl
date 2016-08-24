export h5save, h5load, h5dict, h5convert

"""
    h5save(filename::String, data)

Save objects as a HDF5 format.
Note that the objects are required to implement `h5convert` and `h5load!` functions.
"""
function h5save(filename::String, data)
    function f(g::HDF5Group, dict::Dict)
        for (k,v) in dict
            if typeof(v) <: Dict
                c = g_create(g, k)
                f(c, v)
            else
                g[k] = v
            end
        end
    end

    h5open(filename, "w") do h
        h["version"] = string(VERSION)
        g = g_create(h, "Merlin")
        f(g, h5convert(data))
    end
end

"""
    h5load(filename::String)

Load a HDF5 file.
"""
h5load(filename::String) = h5load!(h5read(filename,"Merlin"))

"""
    h5dict(T::Type, x::Pair...)

Create a hdf5 dictionary with type information.
"""
function h5dict(T::Type, x::Pair...)
    dict = Dict{String,Any}("#TYPE"=>string(T))
    for (k,v) in x
        dict[string(k)] = h5convert(v)
    end
    dict
end

h5convert(x::Number) = x
h5convert{T<:Number}(x::Array{T}) = x
h5convert(x::String) = x
h5convert(x::Symbol) = h5dict(Symbol, "s"=>string(x))
h5convert(x::Function) = h5dict(Function, "f"=>string(x))

function h5convert(x::Vector{Any})
    dict = h5dict(Vector{Any})
    for i = 1:length(x)
        dict[string(i)] = h5convert(x[i])
    end
    dict
end

function h5convert(x::Dict)
    dict = Dict{String,Any}()
    for (k,v) in x
        dict[string(k)] = h5convert(v)
    end
    dict
end

function h5load!(data::Dict)
    if haskey(data, "#TYPE")
        T = eval(parse(data["#TYPE"]))
        delete!(data, "#TYPE")
        h5load!(T, data)
    else
        for (k,v) in data
            typeof(v) <: Dict && (data[k] = h5load!(v))
        end
        data
    end
end

h5load!(::Type{Function}, data) = parse(data["f"])
h5load!(::Type{Symbol}, data) = parse(data["s"])
h5load!(x::Number) = x
h5load!{T<:Number}(x::Array{T}) = x
h5load!(x::String) = x

function h5load!(::Type{Vector{Any}}, data::Dict)
    vec = []
    for (k,v) in data
        i = parse(Int, k)
        while i > length(vec)
            push!(vec, nothing)
        end
        vec[i] = h5load!(v)
    end
    vec
end
