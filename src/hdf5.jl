"""
    save(path::String, mode::String, name::String, obj)

Save an object in Merlin HDF5 format.
* mode: "w" (overrite) or "r+" (append)
"""
function save(path::String, mode::String, name::String, obj)
    mkpath(dirname(path))
    h5open(path, mode) do h
        h5save(h, name, obj)
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

h5convert(x::Symbol) = string(x)
h5convert(::Type{Symbol}, x) = parse(x)

h5convert(x::Void) = string(x)
h5convert(::Type{Void}, x) = nothing

h5convert(x::DataType) = string(x)
h5convert(::Type{DataType}, x) = eval(parse(x))

h5convert(x::Vector) = Dict(i=>x[i] for i=1:length(x))
function h5convert{T<:Vector}(::Type{T}, x::Dict)
    vec = Array(eltype(T), length(x))
    for (k,v) in x
        vec[parse(Int,k)] = v
    end
    vec
end

h5convert(x::Tuple) = h5convert([x...])
h5convert{T<:Tuple}(::Type{T}, x) = tuple(h5load(Vector,x)...)

h5convert(x::Function) = throw("Saving a function object is not supported. Override `h5convert`.")

h5convert(x) = Dict(name=>getfield(x, name) for name in fieldnames(x))
function h5convert(T, x)
    values = map(name -> x[string(name)], fieldnames(T))
    T(values...)
end

function h5save{T}(group, key::String, obj::T)
    if T <: HDF5.BitsKindOrString ||
        (T <: Array && eltype(obj) <: HDF5.BitsKindOrString)
        group[key] = obj
    elseif T <: Function
        h5save(group, key, Symbol(obj))
    else
        h5obj = h5convert(obj)
        if typeof(h5obj) <: Dict
            g = g_create(group, key)
            attrs(g)["#JULIA_TYPE"] = string(T)
            for (k,v) in h5obj
                h5save(g, string(k), v)
            end
        else
            group[key] = h5obj
            attrs(group[key])["#JULIA_TYPE"] = string(T)
        end
    end
end

function h5load(group::HDF5Group)
    dict = Dict()
    for name in names(group)
        dict[name] = h5load(group[name])
    end
    attr = read(attrs(group), "#JULIA_TYPE")
    T = eval(parse(attr))
    h5convert(T, dict)
end

function h5load(dataset::HDF5Dataset)
    data = read(dataset)
    if exists(attrs(dataset), "#JULIA_TYPE")
        attr = read(attrs(dataset), "#JULIA_TYPE")
        T = eval(parse(attr))
        h5convert(T, data)
    else
        data
    end
end
