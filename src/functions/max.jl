import Base.max

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
function max(x::Var, dim::Int)
    x.data == nothing && return Var(nothing, (max,x,dim))
    # TODO: in-place findmax
    data, idx = findmax(x.data, dim)
    y = Var(data, (x,))
    y.df = () -> isconst(x) || ∇max!(y.grad, idx, x.grad)
    y
end

function ∇max!{T}(gy::Array{T}, idx::Array{Int}, gx::Array{T})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
