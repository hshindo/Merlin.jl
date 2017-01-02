import Base.reshape
export reshape4d

"""
    reshape(x::Var, dims::Int...)
"""
function reshape(x::Var, dims::Tuple)
    isvoid(x.data) && return Var(nothing, reshape, (x,dims))
    y = reshape(x.data, dims)
    df(gy) = isvoid(x.grad) || BLAS.axpy!(eltype(gy)(1), gy, x.grad)
    Var(y, df, (x,))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)

function reshape4d(x::Var)
    isvoid(x.data) && return Var(nothing, reshape4d, (x,))
    y = reshape4d(x.data)
    df(gy) = isvoid(x.grad) || BLAS.axpy!(eltype(gy)(1), gy, x.grad)
    Var(y, df, (x,))
end

reshape4d{T}(x::Array{T,3}) = reshape(x, size(x)..., 1)
