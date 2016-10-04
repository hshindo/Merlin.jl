import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
@graph function reshape(x::Var, dims::Tuple{Vararg{Int}})
    y = reshape(x.data, dims)
    df{T}(gy::UniArray{T}) = isconst(x) || BLAS.axpy!(T(1), gy, x.grad)
    Var(y, [x], df)
end
reshape(x::Var, dims::Int...) = reshape(x, dims)
