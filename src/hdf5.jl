export save_hdf5

type HDF5Dict
    data::Dict
    T::Type
end

#Base.getindex(d::HDF5Dict, key::String) = d.data[key]
#Base.setindex!(d::HDF5Dict, value, key::String) = d.data[key] = value

h5dict(T::Type, x::Pair...) = Dict{Any,Any}("JULIA_TYPE" => T, x...)

to_hdf5(x::Number) = x
to_hdf5{T<:Number}(x::Array{T}) = x
to_hdf5(x::String) = x
to_hdf5(x::Symbol) = h5dict(Symbol, "s"=>string(x))
to_hdf5(x::Function) = h5dict(Function, "f"=>string(x))
to_hdf5(x::DataType) = h5dict(DataType, "t"=>string(x))
to_hdf5(x) = h5dict(typeof(x), "t"=>string(x))

function to_hdf5(x::Vector{Any})
    dict = h5dict(Vector{Any})
    for i = 1:length(x)
        dict[i] = to_hdf5(x[i])
    end
    dict
end

function to_hdf5(x::Dict)
    dict = Dict{String,Any}()
    for (k,v) in x
        dict[string(k)] = h5convert(v)
    end
    dict
end

function save_hdf5(filename::String, key::String, obj)
    function write(g::HDF5Group, obj)
        attrs(g)["JULIA_TYPE"] = string(typeof(obj))
        dict = to_hdf5(obj)
        for (k,v) in dict
            if isnative(v)
                g[k] = v
            else
                write(g_create(g,string(k)), v)
            end
        end
    end

    dict = to_hdf5(obj)
    h5open(filename, "w") do h
        g = g_create(h, key)
        write(g, dict)
    end
end

function h5write()
end

function h5writedict(g, data::Dict)
    for (k,v) in data
        if typeof(v) <: Dict
            c = g_create(g, string(k))
            h5writedict(c, v)
        else
            g[string(k)] = v
        end
    end
end
