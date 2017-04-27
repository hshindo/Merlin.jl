import Base: max, sum

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
max(x::Var, dim::Int) = Max(dim)(x)

type Max
    dim::Int
end

function (f::Max)(x::Var)
    y = Var(nothing, f, (x,))
    y.data, idx = findmax(x.data, f.dim)
    y.df! = function df!()
        isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
    end
    y
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
sum(x::Var, dim::Int) = Sum(dim)(x)

type Sum
    dim::Int
end

function (f::Sum)(x::Var)
    y = Var(sum(x.data,f.dim), f, (x,))
    y.df! = function df!()
        isvoid(x.grad) || broadcast!(+, x.grad, x.grad, y.grad)
    end
    y
end
