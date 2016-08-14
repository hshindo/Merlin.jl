export Embedding

type Embedding <: Functor
    ws::Vector{Var}
    idset::IntSet
end

"""
    Embedding{T}(::Type{T}, indim, outdim)

### 👉 Example
```julia
f = Embedding(Float32,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
function Embedding{T}(::Type{T}, indim::Int, outdim::Int)
    ws = Var[Param(Vector{T}(randn(outdim))) for i=1:indim]
    Embedding(ws, IntSet())
end

"""
    Embed(path, T)

Construc embeddings from file.
"""
function Embedding(path, T::Type)
    lines = open(readlines, path)
    ws = Array(Var, length(lines))
    for i = 1:length(lines)
        items = split(chomp(lines[i]), ' ')
        w = map(x -> parse(T,x), items)
        ws[i] = Param(w)
    end
    Embedding(ws, IntSet())
end

@compat function (f::Embedding)(x::Var)
    y = embedding(f.ws, x.data)
    function df(gy)
        ∇embedding!(f.ws, x.data, gy)
        for id in x.data
            push!(f.idset, id)
        end
    end
    Var(y, [x], f, df)
end

@compat (f::Embedding)(x::GraphNode) = GraphNode(f, x)

function embedding(ws::Vector{Var}, x::Array{Int})
    n = length(ws[1].data)
    dims = [size(x)...]
    dims[1] *= n
    y = similar(ws[1].data, dims...)
    for i = 1:length(x)
        copy!(y, (i-1)*n+1, ws[x[i]].data, 1, n)
    end
    y
end

function ∇embedding!{T}(ws::Vector{Var}, x::Array{Int}, gy::Array{T})
    n = length(ws[1].data)
    for i = 1:length(x)
        gw = ws[x[i]].grad
        BLAS.axpy!(n, T(1), pointer(gy,(i-1)*n+1), 1, pointer(gw), 1)
    end
end

function update!(f::Embedding, opt)
    for id in f.idset
        w = f.ws[id]
        opt(w.data, w.grad)
    end
    empty!(f.idset)
end

export quantize!
function quantize!(f::Embedding)
    for w in f.ws
        x = w.data
        for i = 1:length(x)
            x[i] = round(x[i], 1)
        end
    end
end
