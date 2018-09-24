function transpose_batch(xs::Vector{Var}, rev::Bool)
    @assert issorted(xs, rev=true, by = x -> size(x,2))
    y = transpose_batch(map(x -> x.data, xs), rev)
    y = Var(y, (transpose_batch,xs,rev))
    split(y, 2, transpose_dims(xs))
end

function transpose_dims(xs::Vector{Var})
    k = length(xs)
    dims = Int[]
    for t = 1:size(xs[1],2)
        while dims[k] < t
            k -= 1
        end
        push!(dims, k)
    end
    dims
end

function transpose_batch(xs::Vector{Matrix{T}}, rev::Bool) where T
    n = size(xs[1], 1)
    y = Array{T}(n, sum(x -> size(x,2), xs))
    yi = 1
    for t = 1:size(xs[1],2)
        for x in xs
            size(x,2) < t && break
            i = rev ? size(x,2)-t+1 : t
            xi = n * (i-1) + 1
            copyto!(y, yi, x, xi, n)
            yi += n
        end
    end
    y
end

function cumsum_cint(dims::Vector{Int})
    cumdims = Array{Cint}(length(dims)+1)
    cumdims[1] = 0
    for i = 2:length(cumdims)
        cumdims[i] = cumdims[i-1] + dims[i-1]
    end
    cumdims
end
