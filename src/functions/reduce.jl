import Base: max, sum

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
max(x::Var, dim::Int) = forward0(max, x, dim)

function forward(::typeof(max), x::Array, dim::Int)
    y, idx = findmax(x, dim)
    backward!(gy, gx) = isvoid(gx) || ∇max!(gy, gx, idx)
    y, backward!
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
sum(x::Var, dim::Int) = forward0(sum, x, dim)

function forward(::typeof(sum), x::Array, dim::Int)
    y = sum(x, dim)
    backward!(gy, gx) = isvoid(gx) || broadcast!(+, gx, gx, gy)
    y, backward!
end
