import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
function reshape(x::Var, dims::Tuple)
    y = reshape(x.data, dims)
    df(gy) = isvoid(x.grad) || BLAS.axpy!(T(1), gy, x.grad)
    Var(y, df, (x,))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)
reshape(x::Var{Void}, dims::Tuple) = Var(Void(), reshape, (x,dims))
