doc"""
    L2(lambda)

L2 regularizer.
"""
mutable struct L2
    lambda
end

function (reg::L2)(data::Array{T}, grad::Array{T}) where T
    BLAS.axpy!(T(reg.lambda), data, grad)
end
