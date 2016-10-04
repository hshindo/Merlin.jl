h5object(x::Symbol) = string(x)
h5object(x::Void) = string(x)
h5object(x::DataType) = string(x)
h5object(x::Vector) = Dict(i=>x[i] for i=1:length(x))
h5object(x::Tuple) = h5object([x...])
h5object(x) = Dict(name=>getfield(x, name) for name in fieldnames(x))

h5load(::Type{Symbol}, x) = parse(x)
h5load(::Type{Void}, x) = nothing
h5load(::Type{DataType}, x) = eval(parse(x))
h5load{T<:Tuple}(::Type{T}, x) = tuple(h5load(Vector,x)...)
function h5load{T<:Vector}(::Type{T}, x::Dict)
    vec = Array(eltype(T), length(x))
    for (k,v) in x
        vec[parse(Int,k)] = v
    end
    vec
end
function h5load(T, x)
    values = map(name -> x[string(name)], fieldnames(T))
    T(values...)
end

function save(path::String, key::String, obj)
    mkpath(dirname(path))
    # Since HDF5 doesn't support 'a' option, emulate it.
    if !isfile(path)
        h5open(path, "w") do h end
    end

    h5open(path, "r+") do h
        h5save(h, key, obj)
    end
    nothing
end

function h5save{T}(group, key::String, obj::T)
    if T <: HDF5.BitsKindOrString ||
        (T <: Array && eltype(obj) <: HDF5.BitsKindOrString)
        group[key] = obj
    elseif T <: Function
        h5save(group, key, Symbol(obj))
    else
        h5obj = h5object(obj)
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

function load(path::String, key::String)
    h5open(path, "r") do h
        h5load(h[key])
    end
end

function h5load(group::HDF5Group)
    dict = Dict()
    for name in names(group)
        dict[name] = h5load(group[name])
    end
    attr = read(attrs(group), "#JULIA_TYPE")
    T = eval(parse(attr))
    h5load(T, dict)
end

function h5load(dataset::HDF5Dataset)
    data = read(dataset)
    if exists(attrs(dataset), "#JULIA_TYPE")
        attr = read(attrs(dataset), "#JULIA_TYPE")
        T = eval(parse(attr))
        h5load(T, data)
    else
        data
    end
end
