export to_hdf5, save_hdf5, load_hdf5

"""
    save_hdf5(path, objs...)

Save objects as a HDF5 file.
"""
function save_hdf5(path::String, objs::Pair...)
    dict = Dict()
    for (k,v) in objs
        dict["$(k)::$(typeof(v))"] = to_hdf5(v)
    end
    h5open(path, "w") do h
        g = g_create(h, "Merlin")
        write_hdf5(g, dict)
    end
end

function write_hdf5(group::HDF5Group, dict::Dict)
    for (k,v) in dict
        if typeof(v) <: Dict
            g = g_create(group, string(k))
            write_hdf5(g, v)
        else
            group[string(k)] = v
        end
    end
end

"""
    load_hdf5

Load HDF5 file.
"""
function load_hdf5(path::String)
    h5 = h5read(path, "Merlin")
    dict = Dict()
    for (k,v) in h5
        name, obj = parse_hdf5(k, v)
        dict[name] = from_hdf5(T, v)
    end
    dict
end

function parse_hdf5(key::String, value)
    args = parse(key).args
    if length(args) == 1
        obj = value
    else
        
    end
    name = args[1]
    T = eval(args[2])
    name, from_hdf5(T, value)
end

to_hdf5(x::Function) = string(x)
to_hdf5(x::Symbol) = string(x)
to_hdf5(x::Number) = x
to_hdf5{T<:Number}(x::Array{T}) = x

function to_dict{T}(x::T)
    dict = Dict()
    names = fieldnames(T)
    for i = 1:nfields(T)
        f = getfield(x, i)
        dict[string(names[i])] = f
    end
    Dict(string(T) => dict)
end
