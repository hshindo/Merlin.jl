"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function Base.max(x::Var, dim::Int)
    y = Var(nothing, nothing, max, (x,dim))
    y.data, idx, y.batchdims = findmax(x.data, dim, x.batchdims)
    y.df! = () -> begin
        isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
    end
    y
end

function Base.findmax(x::Array{T,N}, dim::Int, batchdims::Vector{Int}) where {T,N}
    if dim == N
        if all(d -> d == x.dims[1], batchdims)
            x = reshape(x, Base.front(size(x))..., batchdims[1], length(batchdims))
            y, idx = findmax(x, dim)
            y = reshape(y, size(y)[1:N-1]..., size(y,N+1))
            y, idx, ones(batchdims)
        else
            y = Array{T}(Base.front(size(x))..., length(batchdims))
            idx = Array{Int}(size(y))

            xoffset, yoffset = 0, 0
            n = stride(x, N)
            _ydims = ntuple(i -> i == N ? 1 : size(y,i), N)
            for d in batchdims
                _xdims = ntuple(i -> i == N ? d : size(x,i), N)
                _x = unsafe_wrap(Array, pointer(x,xoffset+1), _xdims)
                _y = unsafe_wrap(Array, pointer(y,yoffset+1), _ydims)
                _idx = unsafe_wrap(Array, pointer(idx,yoffset+1), _ydims)
                findmin!(_y, _idx, _x)
                @inbounds for k = 1:length(_idx)
                    _idx[k] += xoffset
                end
                xoffset += n * d
                yoffset += n
            end
            y, idx, ones(batchdims)
        end
    else
        y, idx = findmax(x.data, dim)
        y, idx, batchdims
    end
end

function ∇max!(gy::Array{T}, gx::Array{T}, idx::Array{Int}) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
