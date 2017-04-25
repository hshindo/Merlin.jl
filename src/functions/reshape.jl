import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
reshape(x::Var, dims::Tuple) = forward0(reshape, x, dims)
reshape(x::Var, dims::Int...) = reshape(x, dims)

function forward{T}(::typeof(reshape), x::UniArray{T}, dims::Tuple)
    y = copy(reshape(x, dims))
    backward!(gy, gx) = isvoid(gx) || BLAS.axpy!(T(1), gx, gy)
    y, backward!
end
