import Base: max, sum

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function max(x::Var, dim::Int)
    isvoid(x.data) && return Var(nothing, max, (x,dim))
    iscuda(x.data) && return CUDA.max(x, dim)
    y, idx = findmax(x.data, dim)
    df(gy) = isvoid(x.grad) || ∇max!(gy, x.grad, idx)
    Var(y, df, (x,))
end

function ∇max!{T}(gy::Array{T}, gx::Array{T}, idx::Array{Int})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

"""
    sum(x::Var, dim::Int)

Returns the sum over the given dimension.
"""
function sum(x::Var, dim::Int)
    isvoid(x.data) && return Var(nothing, sum, (x,dim))
    iscuda(x.data) && return CUDA.sum(x, dim)
    y = sum(x.data, dim)
    df(gy) = isvoid(x.grad) || broadcast!(.+, x.grad, x.grad, gy)
    Var(y, df, (x,))
end
