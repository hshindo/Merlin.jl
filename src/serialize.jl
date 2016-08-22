export HDFDict

type HDFDict
    dict::Dict{String,Any}
end

HDFDict(x::Pair...) = HDFDict(Dict{String,Any}(x))
HDFDict(T::Type) = HDFDict("#julia_type"=>string(T))
HDFDict(x::Number) = HDFDict("#data"=>x)
HDFDict{T<:Number}(x::Array{T}) = HDFDict("#data"=>x)
HDFDict(x::Function) = HDFDict("#data"=>string(x), "#julia_type"=>"Function")
HDFDict(x::Symbol) = HDFDict("#data"=>string(x), "#julia_type"=>"Symbol")

function HDFDict(T::Type, x)
    dict = HDFDict(x)
    dict.dict["#julia_type"] = string(T)
    dict
end

function Base.setindex!(dict::HDFDict, value, key)
    if typeof(value) <: HDFDict
        dict.dict[string(key)] = value
    else
        dict.dict[string(key)] = HDFDict(value)
    end
end

function HDFDict(x::Vector)
    dict = HDFDict(Vector)
    for i = 1:length(x)
        dict[i] = x[i]
    end
    dict
end

function HDFDict(x::Graph)
    dict = HDFDict(Graph)
    argdict = ObjectIdDict()
    for i = 1:length(x)
        d = HDFDict(GraphNode)
        dict[i] = d
        for j = 1:length(x[i])
            n = x[i][j]
            if typeof(n) == GraphNode
                d[j] = HDFDict("#julia_type"=>"GraphNode", "#data"=>argdict[n])
            else
                d[j] = n
            end
        end
        argdict[x[i]] = i
    end
    dict
end

function save(path::String, hdf5::HDFDict)
    function write(g, h5::HDFDict)
        for (k,v) in h5.dict
            if typeof(v) <: HDFDict
                gg = g_create(g, k)
                write(gg, v)
            else
                g[k] = v
            end
        end
    end

    h5open(path, "w") do h
        h["version"] = string(VERSION)
        g = g_create(h, "Merlin")
        write(g, hdf5)
    end
end
