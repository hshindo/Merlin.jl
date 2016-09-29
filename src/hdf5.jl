export save_hdf5

to_hdf5(x::Vector{Any}) = Dict(i => x[i] for i=1:length(x))
to_hdf5(x::Dict) = x
to_hdf5(x::Symbol) = string(x)
to_hdf5(x::Function) = string(x)
to_hdf5(x::DataType) = string(x)
to_hdf5(x::Var) = string(x)

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
            attrs(group[key])["JULIA_TYPE"] = string(T)
            group[key] = h5obj
        end
    end
end

function save_hdf5(filename::String, key::String, obj)
    h5open(filename, "w") do h
        write_hdf5(h, key, obj)
    end
    nothing
end
