import Base.max

"""
    max(x::Var, dim::Int)
Compute the maximum value along the given dimensions.
"""
function max(x::Var, dim::Int)
    x.data == nothing && return Var(nothing, max, (x,dim))
    y, idx = findmax(x.data, dim)
    df(gy) = isconst(x) || ∇max!(gy, idx, x.grad)
    Var(y, max, (x,), df)
end

function ∇max!{T}(gy::Array{T}, idx::Array{Int}, gx::Array{T})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
