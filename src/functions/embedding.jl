export Embedding

type Embedding <: Functor
    ws::Vector{Var}
    idset::IntSet
end

Embedding(ws::Vector{Var}) = Embedding(ws, IntSet())

function Embedding(w::Matrix)
    n = size(w,1)
    ws = Array(Var, size(w,2))
    for i = 1:length(ws)
        ws[i] = Var(w[(i-1)*n+1:i*n])
    end
    Embedding(ws)
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
function Embedding(T::Type, indim::Int, outdim::Int)
    ws = Var[Var(rand(T,outdim)) for i=1:indim]
    Embedding(ws)
end

"""
    Embed(path, T)

Construct embeddings from file.
"""
function Embedding(path, T::Type)
    lines = open(readlines, path)
    ws = Array(Var, length(lines))
    for i = 1:length(lines)
        items = split(chomp(lines[i]), ' ')
        w = map(x -> parse(T,x), items)
        ws[i] = Var(w)
    end
    Embedding(ws)
end

function (f::Embedding)(x::Var)
    y = embedding(f.ws, x.data)
    function df(gy)
        ∇embedding!(f.ws, x.data, gy)
        for id in x.data
            id > 0 && push!(f.idset, id)
        end
    end
    Var(y, [x], f, df)
end

function embedding(ws::Vector{Var}, x::Array{Int})
    T = eltype(ws[1].data)
    n = length(ws[1].data)
    dims = [size(x)...]
    dims[1] *= n
    y = similar(ws[1].data, dims...)
    for i = 1:length(x)
        yi = (i-1) * n + 1
        if x[i] == 0
            y[yi:yi+n-1] = T(0)
        else
            copy!(y, yi, ws[x[i]].data, 1, n)
        end
    end
    y
end

function ∇embedding!{T}(ws::Vector{Var}, x::Array{Int}, gy::Array{T})
    n = length(ws[1].data)
    for i = 1:length(x)
        x[i] == 0 && continue
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

function h5convert(f::Embedding)
    n = length(f.ws[1].data)
    w = similar(f.ws[1].data, length(f.ws[1].data), length(f.ws))
    for i = 1:length(f.ws)
        copy!(w, (i-1)*n+1, f.ws[i].data, 1, n)
    end
    h5dict(Embedding, "w"=>w)
end

h5load!(::Type{Embedding}, data) = Embedding(data["w"])

export quantize!
function quantize!(f::Embedding)
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
