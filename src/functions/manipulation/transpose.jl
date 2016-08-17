import Base: transpose

"""
    transpose(x::Var)
"""
function transpose(x::Var)
    y = transpose(x.data)
    df{T}(gy::UniArray{T}) = hasgrad(x) && BLAS.axpy!(T(1), transpose(gy), x.grad)
    Var(y, [x], transpose, df)
end
