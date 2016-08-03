export Embed

type Embed
    ws::Vector{Var}
end

"""
    Embed{T}(::Type{T}, indim, outdim)
    
### 👉 Example
```julia
f = Embed(Float32,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
function Embed{T}(::Type{T}, indim::Int, outdim::Int)
    ws = Var[Param(Vector{T}(randn(outdim))) for i=1:indim]
    Embed(ws)
end

#=
function Lookup{T}(path, ::Type{T})
    lines = open(readlines, path)
    ws = Array(Var, length(lines))
    for i = 1:length(lines)
        items = split(chomp(lines[i]), ' ')
        w = map(x -> parse(T,x), items)
        ws[i] = param(w)
    end
    Lookup(ws)
end
=#

@compat function (f::Embed)(x::Var)
    ws = f.ws
    ids = x.data
    args = Var[]
    vars = map(id -> ws[id], ids)
    for id in IntSet(ids)
        push!(args, ws[id])
    end
    y = embed(ws, ids)
    df(gy) = ∇embed!(ws, ids, gy)
    Var(y, args, df)
end

function embed(ws::Vector{Var}, x::Array{Int})
    T = eltype(ws[1].data)
    n = length(ws[1].data)
    y = similar(ws[1].data, size(x,1)*n, size(x)[2:end]...)
    for i = 1:length(x)
        copy!(y, (i-1)*n+1, ws[x[i]].data, 1, n)
    end
    y
end

function ∇embed!{T}(ws::Vector{Var}, x::Array{Int}, gy::Array{T})
    n = length(ws[1].data)
    offset = 1
    for i = 1:length(x)
        gw = ws[x[i]].grad
        BLAS.axpy!(n, T(1), pointer(gy,offset), stride(gy,1), pointer(gw), stride(gw,1))
        offset += n
    end
end

function ∇lookup2!{T}(w, gw, x::Array{Int}, gy::Array{T})
    n = size(w, 1)
    for i = 1:length(x)
        soffs = (i - 1) * n + 1
        doffs = (x[i] - 1) * n + 1
        BLAS.axpy!(n, T(1), pointer(gy,soffs), stride(gy,1), pointer(gw,doffs), stride(gw,1))
    end
end
