import Base.max

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
function max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    df(gy) = hasgrad(x) && ∇max!(idx, x.grad, gy)
    Max(y, [x], max, df)
end

max(x::GraphNode, dim::Int) = GraphNode(max, x, dim)

function ∇max!{T}(idx::Vector{Int}, gx::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
