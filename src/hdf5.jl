using HDF5

"""
    save(path::String, didc::Dict, [mode="w"])

Save an object in Merlin HDF5 format.
* mode: "w" (overrite) or "r+" (append)
"""
function save(path::String, dict::Dict; mode="w")
    h5open(path, mode) do h
        h5save(h, dict)
    end
    nothing
end

"""
    load(path::String, name::String)

Load an object from Merlin HDF5 format.
"""
function load(path::String)
    h5open(path, "r") do h
        dict = Dict{String,Any}()
        for name in names(h)
            dict[name] = h5load(h[name])
        end
        dict
    end
end

function h5save(group, dict::Dict)
    for (k,v) in dict
        if isa(v, Union{Real,String})
            group[k] = v
        elseif isa(v, Array) && eltype(v) <: Real
            group[k] = v
        elseif isa(v, Function)
            group[k] = string(v)
            attrs(group[k])["JULIA_TYPE"] = "Function"
        else
            conv = writeas(v)
            if isa(conv, Dict)
                h5save(g_create(group,k), conv)
            else
                group[k] = conv
            end
            attrs(group[k])["JULIA_TYPE"] = string(typeof(v))
        end
    end
end

function h5load(group::HDF5Group)
    dict = Dict()
    for name in names(group)
        dict[name] = h5load(group[name])
    end
    attr = read(attrs(group), "JULIA_TYPE")
    T = eval(current_module(), parse(attr))
    readas(T, dict)
end

function h5load(dataset::HDF5Dataset)
    data = read(dataset)
    if exists(attrs(dataset), "JULIA_TYPE")
        attr = read(attrs(dataset), "JULIA_TYPE")
        T = eval(current_module(), parse(attr))
        readas(T, data)
    else
        data
    end
end

writeas(x::Union{Void,Char,Symbol,DataType}) = string(x)
writeas(x::Union{Tuple,Vector}) = Dict(string(i)=>x[i] for i=1:length(x))

readas(::Type{T}, x::String) where T<:Union{Void,DataType,Function} = eval(parse(x))
readas(::Type{Char}, x::String) = x[1]
readas(::Type{Symbol}, x::String) = parse(x)
function readas(::Type{T}, d::Dict) where T<:Tuple
    data = Array{Any}(length(d))
    for (k,v) in d
        data[parse(Int,k)] = v
    end
    tuple(data...)
end

#=
function readas(T::Type, x)
    values = map(name -> x[string(name)], fieldnames(T))
    T(values...)
end
writeas(x) = Dict(name=>getfield(x,name) for name in fieldnames(x))
=#
