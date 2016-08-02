export Embed

type Embed
    ws::Vector{Var}
    idset::IntSet
end

"""
    Embed{T}(::Type{T}, indim, outdim)

* indim: input dimension
* outdim: output dimension

### 👉 Example
```julia
f = Embed(Float32,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
function Embed{T}(::Type{T}, indim::Int, outdim::Int)
    ws = Var[Param(Vector{T}(randn(outdim))) for i=1:indim]
    Embed(ws, IntSet())
end

"""
    Embed(path, T)

Construc embeddings from file.
"""
function Embed(path, T::Type)
    lines = open(readlines, path)
    ws = Array(Var, length(lines))
    for i = 1:length(lines)
        items = split(chomp(lines[i]), ' ')
        w = map(x -> parse(T,x), items)
        ws[i] = param(w)
    end
    Embed(ws, IntSet())
end

@compat function (v::Embed)(x::Var)
    y = v(x.data)
    df(gy) = ∇embed!(v, x.data, gy)
    Var(y, [v], df)
end

@compat function (v::Embed)(x::Array{Int})
    w1 = v.ws[1]
    T = eltype(w1.data)
    n = length(w1.data)
    dims = [size(x)...]
    dims[1] *= n
    y = similar(x, dims...)
    for i = 1:length(x)
        copy!(y, (i-1)*n+1, v.ws[x[i]].data, 1, n)
    end
    y
end

function ∇embed!{T}(v::Embed, x::Array{Int}, gy::Array{T})
    n = length(v.ws[1].data)
    for i = 1:length(x)
        gw = v.ws[x[i]].grad
        BLAS.axpy!(n, T(1), pointer(gy,(i-1)*n+1), 1, pointer(gw), 1)
        push!(v.idset, x[i])
    end
end

function update!(v::Embed, opt)
    for id in v.idset
        opt(v.ws[id].data, v.ws[id].grad)
    end
    empty!(v.idset)
end

function ∇lookup2!{T}(w, gw, x::Array{Int}, gy::Array{T})
    n = size(w, 1)
    for i = 1:length(x)
        soffs = (i - 1) * n + 1
        doffs = (x[i] - 1) * n + 1
        BLAS.axpy!(n, T(1), pointer(gy,soffs), stride(gy,1), pointer(gw,doffs), stride(gw,1))
    end
end
