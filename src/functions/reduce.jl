import Base: max, sum

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
function max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    df(gy) = isvoid(x.grad) || ∇max!(gy, x.grad, idx)
    Var(y, df, (x,))
end
max(x::Var{Void}, dim::Int) = Var(Void(), max, (x,dim))

function ∇max!{T}(gy::Array{T}, gx::Array{T}, idx::Array{Int})
    @inbounds @simd for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

"""
    sum(x::Var, dim::Int)

Compute the sum along the given dimension.
"""
function sum(x::Var, dim::Int)
    y = sum(x.data, dim)
    df(gy) = isvoid(x.grad) || broadcast!(.+, x.grad, x.grad, gy)
    Var(y, df, (x,))
end
sum(x::Var{Void}, dim::Int) = Var(Void(), sum, (x,dim))
