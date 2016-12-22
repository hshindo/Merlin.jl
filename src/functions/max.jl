import Base.max

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
function max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    df(gy) = isvoid(x.grad) || ∇max!(gy, idx, x.grad)
    Var(y, df, (x,))
end
max(x::Var{Void}, dim::Int) = Var(Void(), max, (x,dim))

function ∇max!{T}(gy::Array{T}, idx::Array{Int}, gx::Array{T})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
