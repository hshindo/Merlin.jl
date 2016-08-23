export h5convert, h5save, h5load

"""
    h5save(filename::String, data)

Save objects as a HDF5 format.
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
h5load(filename::String) = h5deconvert(h5read(filename,"Merlin"))

function h5convert(T::Type, x::Pair...)
    dict = h5convert(x...)
    dict["#TYPE"] = string(T)
    dict
end

h5convert(x::Number) = x
h5convert{T<:Number}(x::Array{T}) = x
h5convert(x::String) = x
h5convert(x::Symbol) = h5convert(Symbol, "#NAME"=>string(x))
h5convert(x::Function) = h5convert(Function, "#NAME"=>string(x))

function h5convert(x::Vector{Any})
    dict = h5convert(Vector{Any})
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

function h5convert(x::Pair...)
    dict = Dict{String,Any}()
    for (k,v) in x
        dict[string(k)] = h5convert(v)
    end
    dict
end

function h5deconvert(data::Dict)
    if haskey(data, "#TYPE")
        T = eval(parse(data["#TYPE"]))
        delete!(data, "#TYPE")
        h5deconvert(T, data)
    else
        for (k,v) in data
            typeof(v) <: Dict && (data[k] = f(v))
        end
        data
    end
end

h5deconvert(::Type{Function}, data) = parse(data["#NAME"])
h5deconvert(::Type{Symbol}, data) = parse(data["#NAME"])
h5deconvert(x::Number) = x
h5deconvert{T<:Number}(x::Array{T}) = x
h5deconvert(x::String) = x

function h5deconvert(::Type{Vector{Any}}, dict::Dict)
    vec = []
    for (k,v) in dict
        i = parse(Int, k)
        while i > length(vec)
            push!(vec, nothing)
        end
        vec[i] = h5deconvert(v)
    end
    vec
end

function h5convert(x::Graph)
    dict = h5convert(Graph)
    argdict = ObjectIdDict()
    for i = 1:length(x)
        d = h5convert(GraphNode)
        dict[string(i)] = d
        for j = 1:length(x[i])
            n = x[i][j]
            if typeof(n) == GraphNode
                d[string(j)] = h5convert(GraphNode)
                d[string(j)]["id"] = argdict[n]
            else
                d[string(j)] = h5convert(n)
            end
        end
        argdict[x[i]] = i
    end
    dict
end

function h5deconvert(::Type{Graph}, data::Dict)
    nodes = GraphNode[]
    for (k,v) in data

    end
end
