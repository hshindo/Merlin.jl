export embeddings, lookup

function embeddings(::Type{T}, insize::Int, outsize::Int; init_W=Normal(0,0.01)) where T
    W = init_W(T, outsize, insize)
    [zerograd(W[:,i]) for i=1:size(W,2)]
end

function lookup(embeds::Vector{Var}, x::Var)
    isvoid(x.data) && return Var(nothing,(lookup,embeds,x))
    y = lookup(embeds, x.data)
    xs = map(i -> embeds[i], vec(x.data))
    Var(y, (lookup,xs))
end

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
