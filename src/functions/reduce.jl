function Base.findmax(x::BatchedArray{T,N}, dim::Int) where {T,N}
    if dim == N
        if isconstsize(x)
            ydims = ntuple(i -> i == dim ? batchsize(x) : size(x,i), N)
            y = Array{T}(ydims)
            idx = Array{Int}(ydims)

            offset = 1
            yydims = ntuple(i -> i == N ? 1 : size(y,i), N)
            for d in x.dims
                xxdims = ntuple(i -> i == N ? d : size(x,i), N)
                xx = unsafe_wrap(Array, pointer(x.data,offset), xxdims)
                yy = unsafe_wrap(Array, pointer(y,offset), yydims)
                ii = unsafe_wrap(Array, pointer(idxs,offset), yydims)
                findmax2(>, yy, ii, xx)
                offset += d
            end
            BatchedArray(y,ones(x.dims)), idx
        else
            xx = reshape(x.data, Base.front(size(x))..., x.dims[1], batchsize(x))
            y, idx = findmax(xx, dim)
            y = reshape(y, Base.front(size(y)))
            BatchedArray(y,ones(x.dims)), idx
        end
    else
        y, idx = findmax(x.data, dim)
        BatchedArray(y,x.size), idx
    end
end

function Base.sum(x::BatchedArray{T,N}, dim::Int) where {T,N}
    if dim == N

    else
        
    end
end
