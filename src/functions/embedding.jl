export Embedding

doc"""
    Embedding{T}(::Type{T}, insize::Int, outsize::Int)

* insize: input size
* outsize: output size

```julia
T = Float32
f = Embedding(T,10000,100) # 100-length vector, 10k vocabulary
x = Var(rand(1:1000,5,3))
y = f(x)
```
"""
struct Embedding <: AbstractVar
    w::Var
    idset::IntSet
end

function Embedding{T}(w::Matrix{T}; fixed=false)
    w = fixed ? Var(w) : zerograd(w)
    Embedding(w, IntSet())
end

function Embedding{T}(::Type{T}, insize::Int, outsize::Int; fixed=false, init=Normal(0,0.01))
    Embedding(w, fixed=fixed)
end

function lookup(e::Embedding, x::Var)
    y = lookup(e.w.data, x.data)
    Var(y, x.batchdims, lookup, (x,))
end

lookup(e::Embedding, x::Node; name="lookup") = Node(lookup, e, x, name=name)

function lookup{T}(w::Matrix{T}, x::Array{Int})
    n = size(w, 1)
    y = Array{T}(n, size(x)...)
    for i = 1:length(x)
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        copy!(y, yi, w, wi, n)
    end
    y
end

function addgrad!(y::Var, ::typeof(lookup), e::Embedding, x::Var)
    if !isvoid(e.w.grad)
        ∇lookup!(y.grad, e.w.grad, x.data)
        push!(e.idset, x.data...)
    end
end

function ∇lookup!{T}(gy::Array{T}, gw::Matrix{T}, x::Array{Int})
    n = size(gw, 1)
    for i = 1:length(x)
        pgy = pointer(gy, (i-1)*n+1)
        pgw = pointer(gw, (x[i]-1)*n+1)
        BLAS.axpy!(n, T(1), pgy, 1, pgw, 1)
    end
end

function update!(e::Embedding, opt)
    for id in e.idset

        p = f.params[id]
        opt(p.data, p.grad)
    end
    empty!(f.idset)
end

function Base.convert(::Type{H5Object}, e::Embedding)
    data = map(p -> p.data, f.params)
    data = hcat(data...)
    H5Object(Lookup, data)
end
Base.convert(::Type{Embedding}, o::H5Object) = Lookup(o.data)
