"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function Base.max(x::Var, dim::Int)
    y = Var(nothing, max, (x,dim))
    y.data, idx = findmax(x.data, dim)
    y.df! = () -> begin
        isconst(x) || ∇max!(y.grad, x.grad, idx)
    end
    y
end

function Base.findmax(x::BatchedArray{T,N}, dim::Int) where {T,N}
    if dim == N
        if all(d -> d == x.dims[1], x.dims)
            xx = Array(x)
            y, idx = findmax(xx, dim)
            y = reshape(y, size(y)[1:N-1]..., size(y,N+1))
            y, idx
        else
            ydims = ntuple(i -> i == dim ? batchsize(x) : size(x,i), N)
            y = Array{T}(ydims)
            idx = Array{Int}(ydims)

            xxoffset, yyoffset = 0, 0
            n = stride(x.data, N)
            yydims = ntuple(i -> i == N ? 1 : size(y,i), N)
            for d in x.dims
                xxdims = ntuple(i -> i == N ? d : size(x,i), N)
                xx = unsafe_wrap(Array, pointer(x.data,xxoffset+1), xxdims)
                yy = unsafe_wrap(Array, pointer(y,yyoffset+1), yydims)
                ii = unsafe_wrap(Array, pointer(idx,yyoffset+1), yydims)
                Base.findminmax!(>, yy, ii, xx)
                @inbounds for k = 1:length(ii)
                    ii[k] += xxoffset
                end
                xxoffset += n * d
                yyoffset += n
            end
            y, idx
        end
    else
        y, idx = findmax(x.data, dim)
        BatchedArray(y,x.dims), idx
    end
end

function ∇max!(gy::Array{T}, gx::Array{T}, idx::Array{Int}) where T
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
∇max!(gy::BatchedArray, gx::BatchedArray, idx) = ∇max!(gy.data, gx.data, idx)
∇max!(gy::Array, gx::BatchedArray, idx) = ∇max!(gy, gx.data, idx)

function Base.sum(x::BatchedArray{T,N}, dim::Int) where {T,N}
    if dim == N

    else

    end
end
