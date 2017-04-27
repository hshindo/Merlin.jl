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
    Lookup{T}(::Type{T}, indim::Int, outdim::Int)

* indim: input dimension
* outdim: output dimension

```julia
T = Float32
f = Lookup(T,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
function Lookup{T}(::Type{T}, indim::Int, outdim::Int)
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
        ws[i] = zerograd(w)
    end
    Lookup(ws)
end

function (f::Lookup)(x::Var)
    y = Var(nothing, f, (x,))
    y.data = f(x.data)
    y.df! = function df!()
        ∇lookup!(y.grad, f.ws, x.data)
        for id in x.data
            id > 0 && push!(f.idset, id)
        end
    end
    y
end

function (f::Lookup){T}(x::Array{T})
    ws = f.ws
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

function ∇lookup!{T}(gy::Array{T}, ws::Vector{Var}, x::Array{Int})
    n = length(ws[1].data)
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

readas(::Type{Lookup}, w) = Lookup(w)
function writeas(f::Lookup)
    data = map(w -> w.data, f.ws)
    hcat(data...)
end
