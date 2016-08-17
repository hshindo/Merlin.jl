import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
function reshape(x::Var, dims::Int...)
    y = reshape(x.data, dims)
    df{T}(gy::UniArray{T}) = hasgrad(x) && BLAS.axpy!(T(1), gy, x.grad)
    Var(y, [x], reshape, df)
end
