"""
    IntDict{T}

A dictionary to convert key::T into integer id.

## 👉 Example
```julia
dict = IntDict{String}()
push!(dict, "abc") == 1
push!(dict, "def") == 2
push!(dict, "abc") == 1
dict["abc"] == 1

getkey(dict, id1) == "abc"

count(dict, id1) == 2
```
"""
type IntDict{T}
    key2id::Dict{T,Int}
    id2key::Vector{T}
    id2count::Vector{Int}

    IntDict() = new(Dict{T,Int}(), T[], Int[])
end
IntDict() = IntDict{Any}()

function IntDict(path::String)
    data = map(x -> chomp(x), open(readlines,path))
    IntDict(data)
end

function IntDict{T}(data::Vector{T})
    d = IntDict{T}()
    for x in data
        push!(d, x)
    end
    d
end

Base.count(d::IntDict, id::Int) = d.id2count[id]

Base.getkey(d::IntDict, id::Int) = d.id2key[id]

Base.getindex(d::IntDict, key) = d.key2id[key]

Base.get(d::IntDict, key, default=0) = get(d.key2id, key, default)

Base.length(d::IntDict) = length(d.key2id)

function Base.push!(d::IntDict, key)
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

Base.append!(d::IntDict, keys::Vector) = map(k -> push!(d,k), keys)
