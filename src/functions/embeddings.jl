export embeddings, lookup

function embeddings{T}(mat::Matrix{T}; fixed=false)
    [Var(mat[:,i],fixed=fixed) for i=1:size(mat,2)]
end

function embeddings{T}(::Type{T}, insize::Int, outsize::Int; init_w=Normal(0,0.01))
    w = init_w(T, outsize, insize)
    embeddings(w)
end

function lookup(embeds::Vector{Var}, x::Var)
    y = lookup(embeds, x.data)
    xs = map(i -> embeds[i], vec(x.data))
    Var(y, x.batchdims, lookup, (xs,))
end

lookup(embeds::Vector{Var}, x::Node; name="") = Node(lookup, (embeds,x), name)

function lookup(embeds::Vector{Var}, x::Array{Int})
    e1 = embeds[1].data
    n = length(e1)
    y = similar(e1, n, size(x)...)
    for i = 1:length(x)
        yi = (i-1) * n + 1
        copy!(y, yi, embeds[x[i]].data, 1, n)
    end
    y
end

function addgrad!(y::Var, ::typeof(lookup), xs::Vector{Var})
    T = eltype(y.data)
    n = length(xs[1].data)
    for i = 1:length(xs)
        isvoid(xs[i].grad) && continue
        py = pointer(y.grad, (i-1)*n+1)
        gx = xs[i].grad
        BLAS.axpy!(n, T(1), py, 1, pointer(gx), 1)
    end
end

#=
function Base.convert(::Type{H5Object}, f::Lookup)
    data = map(p -> p.data, f.params)
    data = hcat(data...)
    H5Object(Lookup, data)
end
Base.convert(::Type{Lookup}, o::H5Object) = Lookup(o.data)
=#
