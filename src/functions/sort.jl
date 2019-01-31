function Base.sort(x::Var, dims::Vector{Int}, perm::Vector{Int})
    cumdims = Array{Int}(undef, length(dims)+1)
    cumdims[1] = 1
    for i = 1:length(dims)
        cumdims[i+1] = cumdims[i] + dims[i]
    end

    ydata = similar(x.data)
    xst = stride(x.data, ndims(x))
    yi = 1
    for p in perm
        xi = xst * (cumdims[p]-1) + 1
        n = xst * dims[p]
        copyto!(ydata, yi, x.data, xi, n)
        yi += n
    end
    Var(ydata, ∇sort!, (x,dims,cumdims,perm))
end

function ∇sort!(y::Var, x::Var, dims::Vector{Int}, cumdims::Vector{Int}, perm::Vector{Int})
    xst = stride(x.data, ndims(x))
    yi = 1
    for p in perm
        xi = xst * (cumdims[p]-1) + 1
        n = xst * dims[p]
        copyto!(x.grad, xi, y.grad, yi, n)
        yi += n
    end
end
