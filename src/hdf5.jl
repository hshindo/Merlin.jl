export h5convert

function save(path, key::String, obj::Tuple)
    h5open(path, "w") do h
        write_hdf5(h, key, obj)
    end
    nothing
end

function write_hdf5(group, key::String, obj)
    if typeof(obj) <: Dict
        for (k,v) in obj
            write_hdf5(group, string(k), v)
        end
    elseif typeof(obj) <: Tuple
        write_hdf5(group, key, obj[2])
        println(group[key])
        attrs(group[key])["#JULIA_TYPE"] = string(obj[1])
    else
        group[key] = obj
    end
end

function load()
end

function h5dict(T::Type, data...)
    dict = Dict(k=>v for (k,v) in data)
    dict["#JULIA_TYPE"] = string(T)
    dict
end

h5convert(x::Number) = x
h5convert{T<:Number}(x::Array{T}) = x
h5convert(x::Void) = Void, string(x)
h5convert(x::String) = x
h5convert(x::Tuple) = Tuple, h5convert([x...])
h5convert(x::Dict) = Dict, Dict(k=>h5convert(v) for (k,v) in x)
h5convert(x::Symbol) = Symbol, string(x)
function h5convert(x::Vector{Any})
    d = Dict()
    for i = 1:length(x)
        d[i] = h5convert(x[i])
    end
    typeof(x), d
    #Dict(i=>h5convert(x[i]) for i=1:length(x))
end
