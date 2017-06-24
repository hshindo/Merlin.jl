"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function Base.max(x::Var, dim::Int)
    y = Var(nothing, nothing, (max,x,dim))
    isvoid(x.data) && return y

    if dim == ndims(x.data)
        y.data, idx = findmax2(x.data, x.batchdims)
    else
        y.data, idx = findmax(x.data, dim)
        y.batchdims = x.batchdims
    end
    y.df! = () -> begin
        isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
    end
    y
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
end

type Sum
    dim::Int
end

function (f::Sum)(x::Var)
    y = Var(sum(x.data,f.dim), f, (x,))
    y.df! = function df!()
        isvoid(x.grad) || broadcast!(+, x.grad, x.grad, y.grad)
    end
    y
end
