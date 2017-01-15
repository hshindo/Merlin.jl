import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
reshape(x::Var, dims::Tuple) = forward(reshape, x, dims)
reshape(x::Var, dims::Int...) = reshape(x, dims)

function forward(::typeof(reshape), x::Array, dims::Tuple)
    y = reshape(x, dims)
    backward!(gy, gx, dims) = isvoid(gx) || add!(gx, gy)
    y, backward!
end
