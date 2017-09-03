export Lookup

struct Lookup <: Functor
    params::Vector{Var}
    idset::IntSet
end

function Lookup(mat::Matrix)
    params = Var[zerograd(mat[:,i]) for i=1:size(mat,2)]
    Lookup(params, IntSet())
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
    params = Var[zerograd(rand(T,outsize)) for i=1:insize]
    Lookup(params, IntSet())
end

function (f::Lookup)(x::Var)
    Var(f(x.data), x.batchdims, f, (x,))
end
(f::Lookup)(x::Node) = Node(f, x)

function (f::Lookup)(x::Array{Int})
    p = f.params[1].data
    y = similar(p, size(p)..., size(x)...)
    for i = 1:length(x)
        yi = (i-1) * length(p) + 1
        copy!(y, yi, f.params[x[i]].data, 1, length(p))
    end
    y
end

function addgrad!(y::Var, f::Lookup, x::Var)
    ∇lookup!(y.grad, f, x.data)
    push!(f.idset, x.data...)
end

function ∇lookup!(gy::Array{T}, f::Lookup, x::Array{Int}) where {T}
    p = f.params[1].data
    n = length(p)
    for i = 1:length(x)
        gw = f.params[x[i]].grad
        BLAS.axpy!(n, T(1), pointer(gy,(i-1)*n+1), 1, pointer(gw), 1)
    end
end

function update!(f::Lookup, opt)
    for id in f.idset
        p = f.params[id]
        opt(p.data, p.grad)
    end
    empty!(f.idset)
end

function Base.convert(::Type{H5Object}, f::Lookup)
    data = map(p -> p.data, f.params)
    data = hcat(data...)
    H5Object(Lookup, data)
end
Base.convert(::Type{Lookup}, o::H5Object) = Lookup(o.data)
