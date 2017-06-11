import Base: max, sum

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function max(x::Var, dim::Int)
    y = Var(nothing, max, (x,dim))
    y.data, idx = findmax(x.data, dim)
    y,df! = () -> begin
        isconst(x) || ∇max!(y.grad, x.grad, idx)
    end
    y
end

function Base.findmax{T,N}(x::BatchedArray{T,N}, dim::Int)
    data = Array{T,N-1}[]
    for xx in split(x)
        yy, idx = findmax(xx, dim)
        push!(data, yy)
    end
    BatchedArray(data), []
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
