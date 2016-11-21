export Lookup

type Lookup <: Functor
    ws::Vector{Var}
    idset::IntSet
end

Lookup(ws) = Lookup(ws, IntSet())

function Lookup{T}(w::Matrix{T})
    ws = Array{Var}(size(w,2))
    for i = 1:length(ws)
        ws[i] = zerograd(w[:,i])
    end
    Lookup(ws)
end

"""
    Lookup{T}(::Type{T}, indim, outdim)

### 👉 Example
```julia
f = Lookup(Float32,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
function Lookup(T::Type, indim::Int, outdim::Int)
    ws = Var[zerograd(rand(T,outdim)) for i=1:indim]
    Lookup(ws)
end

"""
    Lookup(path, T)

Construct embeddings from file.
"""
function Lookup(path::String, T::Type)
    lines = open(readlines, path)
    ws = Array(Var, length(lines))
    for i = 1:length(lines)
        items = split(chomp(lines[i]), ' ')
        w = map(x -> parse(T,x), items)
        ws[i] = Var(w)
    end
    Lookup(ws)
end

function (f::Lookup)(x::Var)
    x.data == nothing && return Var(nothing, (f,x))
    w1 = f.ws[1]
    T = eltype(w1)
    dims = ntuple(d -> d==1 ? size(x,d)*length(w1) : size(x,d), ndims(x))
    y = Var(T, dims)
    lookup!(f.ws, x.data, y.data)
    y.df = () -> begin
        ∇lookup!(y.grad, f.ws, x.data)
        for id in x.data
            id > 0 && push!(f.idset, id)
        end
    end
    y
end

function lookup!{T}(ws::Vector{Var}, x::Array{Int}, y::Array{T})
    n = length(ws[1])
    for i = 1:length(x)
        yi = (i-1) * n + 1
        if x[i] == 0
            y[yi:yi+n-1] = T(0)
        else
            copy!(y, yi, ws[x[i]].data, 1, n)
        end
    end
end

function ∇lookup!{T}(gy::Array{T}, ws::Vector{Var}, x::Array{Int})
    n = length(ws[1])
    for i = 1:length(x)
        x[i] == 0 && continue
        gw = ws[x[i]].grad
        BLAS.axpy!(n, T(1), pointer(gy,(i-1)*n+1), 1, pointer(gw), 1)
    end
end

function update!(f::Lookup, opt)
    for id in f.idset
        w = f.ws[id]
        opt(w.data, w.grad)
    end
    empty!(f.idset)
end

function h5convert(f::Lookup)
    data = map(w -> w.data, f.ws)
    hcat(data...)
end

h5convert(::Type{Lookup}, w) = Lookup(w)

export quantize!
function quantize!(f::Lookup)
    for w in f.ws
        x = w.data
        for i = 1:length(x)
            x[i] < -0.0 && (x[i] = 0.0)
            x[1] > 1.0 && (x[i] = 1.0)
            x[i] = round(x[i], 1)
        end
    end
end

### old ###
#=
function (f::Embedding)(x::Var)
    y = embedding(f.w.data, x.data)
    function df(gy)
        ∇embedding!(f.w.data, x.data, gy)
        for id in x.data
            push!(f.idset, id)
        end
    end
    Var(y, [x], f, df)
end

function embedding{T}(w::Array{T}, x::Array{Int})
    n = size(w, 1)
    dims = [size(x)...]
    dims[1] *= n
    y = Array(T, dims...)
    for i = 1:length(x)
        copy!(y, (i-1)*n+1, w, (x[i]-1)*n+1, n)
    end
    y
end

function ∇embedding!{T}(gw::Array{T}, x::Array{Int}, gy::Array{T})
    n = size(gw, 1)
    for i = 1:length(x)
        BLAS.axpy!(n, T(1), pointer(gy,(i-1)*n+1), 1, pointer(gw, (x[i]-1)*n+1), 1)
    end
end
=#
