type IdDict{T}
    key2id::Dict{T,Int}
    id2key::Vector{T}
    id2count::Vector{Int}

    IdDict() = new(Dict{T,Int}(), T[], Int[])
end
IdDict() = IdDict{Any}()

function IdDict{T}(data::Vector{T})
    d = IdDict{T}()
    for x in data
        push!(d, x)
    end
    d
end

function IdDict(f, path::String)
    data = map(x -> f(chomp(x)), open(readlines,path))
    IdDict(data)
end
IdDict(path::String) = IdDict(identity, path)

Base.count(d::IdDict, id::Int) = d.id2count[id]

Base.getkey(d::IdDict, id::Int) = d.id2key[id]

Base.getindex(d::IdDict, key) = d.key2id[key]

Base.get(d::IdDict, key, default=0) = get(d.key2id, key, default)

Base.length(d::IdDict) = length(d.key2id)

function Base.push!{T}(d::IdDict{T}, key::T)
    if haskey(d.key2id, key)
        id = d.key2id[key]
        d.id2count[id] += 1
    else
        id = length(d.id2key) + 1
        d.key2id[key] = id
        push!(d.id2key, key)
        push!(d.id2count, 1)
    end
    id
end

Base.append!(d::IdDict, keys::Vector) = map(k -> push!(d,k), keys)
