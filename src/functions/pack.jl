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

@generated function pack(x::CuArray{T,N}, dims) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void pack(Array<$Ct,4> y, Array<$Ct,3> x) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= y.length()) return;

        int ndidxs[4];
        y.ndindex(ndidxs, idx);
        int m = x1.dims[0];
        if (ndidxs[0] < m) y[idx] = x1(ndidxs[0], ndidxs[1], ndidxs[3]);
        else y[idx] = x2(ndidxs[0]-m, ndidxs[2], ndidxs[3]);
    }""")
    quote
        throw("Not implemented yet.")
        y = similar(x1, size(x1,1)+size(x2,1), size(x1,2), size(x2,2), size(x1,3))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, y, x1, x2)
        y
    end
end

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

function unpack(x::Var, dims::Vector{Int})
    ydata = unpack(x.data, dims)
    Var(ydata, ∇unpack!, (x,dims))
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

function ∇unpack!(y::Var, x::Var, dims::Vector{Int})
    isnothing(x.grad) && return
    xst = stride(x.data, ndims(x))
    yst = stride(y.data, ndims(y))
    xi = 1
    yi = 1
    for d in dims
        n = yst * d
        addto!(x.grad, xi, y.grad, yi, n)
        xi += xst
        yi += n
    end
end
