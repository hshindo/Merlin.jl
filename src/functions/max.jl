import Base.max

function findmax{T,N}(x::BatchedArray{T,N}, dim::Int)
    if dim != N
        throw("Not implemented yet.")
        y, inds = findmax(x.data, dim)
        return BatchedArray(y,x.size), inds
    end

    dim1 = 1
    for i = 1:N-1
        dim1 *= size(x,i)
    end
    dim3 = 

    dims = getdim3d(x.data, dim)
    y = Array{T}(dims[1], dims[3])
    inds = Array{Int}(dims[1], dims[3])
    cumsize = 0
    for s in x.size
        for i = 1:dims[1]
            @inbounds for k = 1:dims[3]
                maxv = x[sub2ind(dims,i,1,k)]
                maxj = 1
                @inbounds for j = 1:dims[2]
                    j += cumsize
                    ind = sub2ind(dims, i, j, k)
                    if x[ind] > maxv
                        maxv = x[ind]
                        maxj = j
                    end
                end
                y[i,k] = maxv
                inds[i,k] = maxj
            end
        end
        cumsize += s
    end

    xdims = Int[size(x)...]
    deleteat!(xdims, dim)
    y = reshape(y, xdims...)
    BatchedArray(y)
end

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function max(x::Var, dim::Int)
    y = Var(nothing, max, (x,dim))
    y.data, idx = findmax(x.data, dim)
    y,df! = () -> begin
        isconst(x) || ∇max!(y.grad, x.grad, idx)
    end
    y
end

function Base.findmax{T,N}(x::BatchedArray{T,N}, dim::Int)
    data = Array{T,N-1}[]
    for xx in split(x)
        yy, idx = findmax(xx, dim)
        push!(data, yy)
    end
    BatchedArray(data), []
end

function ∇max!{T}(gy::BatchedArray{T}, gx::BatchedArray{T}, idx::Vector{Int})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
