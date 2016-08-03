import Base.max

"""
max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
function max(x::Var, dim::Int)
    hasdata(x) || return Max(nothing, nothing, [x], dim)
    y, idx = findmax(x.data, dim)
    Max(y, nothing, [x], dim, nothing)
end

type Max <: Var
    data
    grad
    tails::Vector
    dim::Int
    idx
end

@compat (m::Max)(x::Var) = max(x, m.dim)

function backward!(m::Max)
    hasgrad(m[1]) || return
    ∇max!(m.idx, m[1].grad, m.grad)
end

function ∇max!{T}(idx::Vector{Int}, gx::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
