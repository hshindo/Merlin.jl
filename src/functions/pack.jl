export pack, unpack

function pack(x::Var, dims::Vector{Int}, padding)
    @assert sum(dims) == size(x,ndims(x))
    s = Base.setindex(size(x), maximum(dims), ndims(x))
    ydata = similar(x.data, s..., length(dims))
    fill!(ydata, padding)

    xst = stride(x.data, ndims(x))
    yst = stride(ydata, ndims(ydata))
    xi = 1
    yi = 1
    for d in dims
        n = xst * d
        copyto!(ydata, yi, x.data, xi, n)
        xi += n
        yi += yst
    end
    Var(ydata, ∇pack!, (x,dims))
end
pack(x::Node, dims, padding) = Node(pack, (x,dims,padding))

function ∇pack!(y::Var, x::Var, dims::Vector{Int})
    isnothing(x.grad) && return
    xst = stride(x.data, ndims(x))
    yst = stride(y.data, ndims(y))
    xi = 1
    yi = 1
    for d in dims
        n = xst * d
        addto!(x.grad, xi, y.grad, yi, n)
        xi += n
        yi += yst
    end
end

function pack(x::UniArray{T,N}, dims::Vector{Int}, padding) where {T,N}
    @assert sum(dims) == size(x,N)
    s = Base.setindex(size(x), maximum(dims), N)
    y = similar(x, s..., length(dims))
    fill!(y, padding)

    xst = stride(x, N)
    yst = stride(y, N+1)
    xi = 1
    yi = 1
    for d in dims
        n = xst * d
        copyto!(y, yi, x, xi, n)
        xi += n
        yi += yst
    end
    y
end

function unpack(x::UniArray{T,N}, dims::Vector{Int}) where {T,N}
    @assert length(dims) == size(x,N)
    s = Base.setindex(Base.front(size(x)), sum(dims), N-1)
    y = similar(x, s...)

    xst = stride(x, N)
    yst = stride(y, N-1)
    xi = 1
    yi = 1
    for d in dims
        n = yst * d
        copyto!(y, yi, x, xi, n)
        xi += xst
        yi += n
    end
    y
end
