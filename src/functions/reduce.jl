
function Base.findmax{T,N}(x::BatchedArray{T,N}, dim::Int)
    if dim != N
        #y, idxs = findmax(x.data, dim)
        #return BatchedArray(y,x.size), idxs
    else
        xx = x.data
        r = Base.reducedim_initarray0(xx, dim, typemin(T))
        println(size(r))
        rr = similar(dims->zeros(Int,dims), Base.reduced_indices0(x,dim))
        Base.findminmax!(>, r, rr, x)
        return
    end
    dim3d = size3d(x.data, dim)
    if dim == N
        ydims = ntuple(i -> i == dim ? batchsize(x) : size(x,i), N)
        y = Array{T}(ydims)
        idxs = Array{Int}(ydims)
        cumsize = 0
        cumidx = 1
        for s in x.size
            for i = 1:dim3d[1]
                for k = 1:dim3d[3]
                    maxx = x[sub2ind(dim3d,i,1,k)]
                    maxj = 1
                    for j = 1:s
                        idx = sub2ind(dim3d, i, j+cumsize, k)
                        if x[idx] > maxx
                            maxx = x[idx]
                            maxj = j + cumsize
                        end
                    end
                    y[cumidx] = maxx
                    idxs[cumidx] = maxj
                    cumidx += 1
                end
            end
            cumsize += s
        end
        BatchedArray(y,ones(x.size)), idxs
    else
        ydims = ntuple(i -> i == dim ? 1 : size(x,i), N)
        y = Array{T}(ydims)
        idxs = Array{Int}(ydims)

        @inbounds for i = 1:dim3d[1]
            @inbounds for k = 1:dim3d[3]
                maxx = x[sub2ind(dim3d,i,1,k)]
                maxj = 1
                @inbounds for j = 1:dim3d[2]
                    idx = sub2ind(dim3d, i, j, k)
                    if x[idx] > maxx
                        maxx = x[idx]
                        maxj = j
                    end
                end
                y[i,k] = maxx
                idxs[i,k] = maxj
            end
        end
        BatchedArray(y,x.size), idxs
    end
end
