export pack, unpack

function pack(x::UniArray, batchdims::Vector{Int}, padding)
    @assert sum(batchdims) == size(x,ndims(x))
    s = Base.setindex(size(x), maximum(batchdims), ndims(x))
    y = similar(x, s..., length(batchdims))
    fill!(y, padding)

    xst = stride(x, ndims(x))
    yst = stride(y, ndims(y))
    xi = 1
    yi = 1
    for d in batchdims
        n = xst * d
        copyto!(y, yi, x, xi, n)
        xi += n
        yi += yst
    end
    y
end

function pack(x::Var, batchdims::Vector{Int}, padding)
    @assert sum(batchdims) == size(x,ndims(x))
    s = Base.setindex(size(x), maximum(batchdims), ndims(x))
    ydata = similar(x.data, s..., length(batchdims))
    fill!(ydata, padding)
    y = Var(ydata, (pack,x,batchdims))

    xst = stride(x, ndims(x))
    yst = stride(y, ndims(y))
    xi = 1
    yi = 1
    for d in batchdims
        n = xst * d
        copyto!(y.data, yi, x.data, xi, n)
        xi += n
        yi += yst
    end
    y
end

function addgrad!(y::Var, ::typeof(pack), x::Var, batchdims::Vector{Int})
    isvoid(x.grad) && return
    xst = stride(x, ndims(x))
    yst = stride(y, ndims(y))
    xi = 1
    yi = 1
    for d in batchdims
        n = xst * d
        addto!(x.grad, xi, y.grad, yi, n)
        xi += n
        yi += yst
    end
end

function unpack(x::UniArray{T,N}, batchdims::Vector{Int}) where {T,N}
    @assert length(batchdims) == size(x,N)
    s = Base.setindex(Base.front(size(x)), sum(batchdims), N-1)
    y = similar(x, s...)

    xst = stride(x, ndims(x))
    yst = stride(y, ndims(y))
    xi = 1
    yi = 1
    for d in batchdims
        n = yst * d
        copyto!(y, yi, x, xi, n)
        xi += xst
        yi += n
    end
    y
end
