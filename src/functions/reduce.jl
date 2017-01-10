import Base: max, sum

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function max(x::Var, dim::Int)
    isa(x.data, Void) && return Var(nothing, max, (x,dim))
    y, idx = findmax(x.data, dim)
    df(gy) = isa(x.grad, Void) || ∇max!(gy, x.grad, idx)
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
    isa(x.data, Void) && return Var(nothing, sum, (x,dim))
    y = sum(x.data, dim)
    df(gy) = isa(x.grad, Void) || broadcast!(+, x.grad, x.grad, gy)
    Var(y, df, (x,))
end
