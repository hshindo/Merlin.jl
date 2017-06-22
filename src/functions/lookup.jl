export Lookup

type Lookup <: Functor
    params::Vector{Var}
    idset::IntSet
end

function Lookup(mat::Matrix)
    params = [zerograd(mat[:,i]) for i=1:size(mat,2)]
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
    y = Var(nothing, nothing, (f,x))
    isvoid(x.data) && return y

    y.data = f(x.data)
    y.batchdims = x.batchdims
    y.df! = () -> begin
        ∇lookup!(y.grad, f, x.data)
        append!(f.idset, x.data)
    end
    y
end

function (f::Lookup)(x::Vector{Int})
    p = f.params[1].data
    y = similar(p, size(p)..., size(x)...)
    for i = 1:length(x)
        yi = (i-1) * length(p) + 1
        copy!(y, yi, f.params[x[i]].data, 1, length(p))
    end
    y
end

function ∇lookup!{T}(gy::Array{T}, f::Lookup, x::Array{Int})
    p = f.params[1].data
    for i = 1:length(x)
        gw = f.params[x[i]].grad
        BLAS.axpy!(length(p), T(1), pointer(gy.data,(i-1)*n+1), 1, pointer(gw), 1)
    end
end

function update!(f::Lookup, opt)
    for id in f.idset
        w = f.ws[id]
        opt(w.data, w.grad)
    end
    empty!(f.idset)
end

readas(::Type{Lookup}, params) = Lookup(params, IntSet())
function writeas(f::Lookup)
    data = map(p -> p.data, f.params)
    hcat(data...)
end
