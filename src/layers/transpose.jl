import Base.transpose

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    hasdata(x) || return Transpose(nothing, nothing, [x])
    Transpose(x.data.', nothing, [x])
end

type Transpose <: Var
    data
    grad
    tails::Vector
end

@compat (::Transpose)(x::Var) = x.'

function backward!(y::Transpose)
    hasgrad(y[1]) || return
    BLAS.axpy!(eltype(y.grad)(1), y.grad.', y[1].grad)
end
