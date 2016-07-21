import Base.reshape

"""
    reshape(x::Var, dims::Int...)

Reshape an array according to the given dimensions.
"""
function reshape(x::Var, dims::Tuple)
    y = reshape(x.data, dims)
    df{T}(gy::UniArray{T}) = hasgrad(x) && BLAS.axpy!(T(1), reshape(gy, dims), x.grad)
    Var(y, [x], df)
end
reshape(x::Var, dims::Int...) = reshape(x, dims)
