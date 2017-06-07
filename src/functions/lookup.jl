export Lookup

type Lookup <: Functor
    ws::Vector{Var}
    idset::IntSet
end

Base.getindex(f::Lookup, key::Int) = f.ws[key]

function Lookup(mat::Matrix)
    ws = [parameter(mat[:,i]) for i=1:size(mat,2)]
    Lookup(ws, IntSet())
end

"""
    Lookup{T}(::Type{T}, insize::Int, outsize::Int)

* insize: input size
* outsize: output size

```julia
T = Float32
f = Lookup(T,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
    ws = Var[parameter(rand(T,outsize)) for i=1:insize]
    Lookup(ws, IntSet())
end

function (f::Lookup)(x::Var)
    y = Var(f(x.data), f, (x,))
    y.df! = function df!()
        ∇lookup!(y.grad, f, x.data)
        append!(f.idset, x.data)
    end
    y
end

function lookup{T}(f::Lookup, x::Array{T})
    w1 = f[1].data
    y = similar(w1, size(w1)..., size(x)...)
    for i = 1:length(x)
        yi = (i-1) * n + 1
        copy!(y, yi, f[x[i]].data, 1, length(w1))
    end
    y
end

function ∇lookup!{T}(gy::Array{T}, f::Lookup, x::Array{Int})
    w1 = f[1].data
    for i = 1:length(x)
        gw = f[x[i]].grad
        BLAS.axpy!(length(w1), T(1), pointer(gy,(i-1)*n+1), 1, pointer(gw), 1)
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
