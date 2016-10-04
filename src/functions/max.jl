import Base.max

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
@graph function max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    df(gy) = isconst(x) || ∇max!(idx, x.grad, gy)
    Var(y, [x], df)
end

function ∇max!{T}(idx::Array{Int}, gx::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
