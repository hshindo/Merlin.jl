export max_batch

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function Base.max(x::Var, dim::Int)
    y = Var(nothing, max, (x,dim))
    y.data, idx = findmax(x.data, dim)
    y.df! = () -> begin
        isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
    end
    y
end

function max_batch(x::Var, ranges::Vector)
    y = Var(nothing, max_batch, (x,ranges))
    y.data, idx = findmax_batch(x.data, ranges)
    y.df! = () -> begin
        isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
    end
    y
end

function findmax_batch(x::Array{T,N}, ranges::Vector) where {T,N}
    y = Array{T}(Base.front(size(x))..., length(ranges))
    idx = Array{Int}(size(y))
    incx = stride(x, N)

    for i = 1:length(ranges)
        _x = view(x, :, ranges[i])
        _y = view(y, :, i)
        _idx = view(idx, :, i)
        findmax!(_y, _idx, _x)
        @inbounds for j = 1:length(_idx)
            _idx[j] += incx * (start(ranges[i])-1)
        end
    end
    y, idx
end

function findmax2(x::Array{T,N}, batchdims::Vector{Int}) where {T,N}
    if all(d -> d == batchdims[1], batchdims)
        x = reshape(x, Base.front(size(x))..., batchdims[1], length(batchdims))
        y, idx = findmax(x, N)
        y = reshape(y, size(y)[1:N-1]..., size(y,N+1))
        y, idx
    else
        y = Array{T}(Base.front(size(x))..., length(batchdims))
        idx = Array{Int}(size(y))

        xoffset, yoffset = 0, 0
        n = stride(x, N)
        _ydims = ntuple(i -> i == N ? 1 : size(y,i), N)
        for d in batchdims
            _xdims = ntuple(i -> i == N ? d : size(x,i), N)
            #_x = view(x, :, xoffset+1:xoffset+d)
            #_y = view(y, :, yoffset+1)
            #_idx = view(idx, :, yoffset+1)
            _x = unsafe_wrap(Array, pointer(x,xoffset+1), _xdims)
            _y = unsafe_wrap(Array, pointer(y,yoffset+1), _ydims)
            _idx = unsafe_wrap(Array, pointer(idx,yoffset+1), _ydims)

            findmax!(_y, _idx, _x)
            @inbounds for k = 1:length(_idx)
                _idx[k] += xoffset
            end
            xoffset += d * n
            yoffset += 1 * n
        end
        y, idx
    end
end

function ∇max!(gy::Array{T}, gx::Array{T}, idx::Array{Int}) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

"""
    sum(x::Var, dim::Int)

Returns the sum over the given dimension.
"""
function Base.sum(x::Var, dim::Int)
    y = Var(nothing, sum, (x,dim))
    y.data = sum(x.data, dim)
    y.df! = () -> begin
        isvoid(x.grad) || broadcast!(+, x.grad, x.grad, y.grad)
    end
    y
end
