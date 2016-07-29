import Base: transpose

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    y = x.data.'
    df{T}(gy::UniArray{T}) = hasgrad(x) && BLAS.axpy!(T(1), gy.', x.grad)
    Var(y, [x], df)
end
