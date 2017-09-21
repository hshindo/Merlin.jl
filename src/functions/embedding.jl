export Embedding
export lookup

doc"""
    Embedding{T}(::Type{T}, insize::Int, outsize::Int)

* insize: input size
* outsize: output size

```julia
T = Float32
e = Embedding(T,10000,100) # 100-length vector, 10000 vocabulary
x = Var(rand(1:1000,5,3))
y = lookup(e, x)
```
"""
struct Embedding <: AbstractVar
    w::Var
    idset::IntSet
end

Embedding(w::Var) = Embedding(w, IntSet())

function Embedding{T}(::Type{T}, insize::Int, outsize::Int; init_w=Normal(0,0.01))
    w = init_w(T, outsize, insize)
    Embedding(Var(w,hasgrad=true))
end

function lookup(e::Embedding, x::Var)
    y = lookup(e.w.data, x.data)
    Var(y, x.batchdims, lookup, (e,x))
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
    n = size(e.w, 1)
    for id in e.idset
        p = pointer(e.w, (id-1)*n+1)
        w = unsafe_wrap(Array, p, n)
        p = pointer(e.gw, (id-1)*n+1)
        gw = unsafe_wrap(Array, p, n)
        opt(w, gw)
    end
    empty!(e.idset)
end

#function Base.convert(::Type{H5Object}, e::Embedding)
#    data = map(p -> p.data, f.params)
#    data = hcat(data...)
#    H5Object(Lookup, data)
#end
#Base.convert(::Type{Embedding}, o::H5Object) = Lookup(o.data)
