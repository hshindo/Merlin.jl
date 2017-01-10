import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
function reshape(x::Var, dims::Tuple)
    isa(x.data, Void) && return Var(nothing, reshape, (x,dims))
    y = reshape(x.data, dims)
    df(gy) = isa(x.grad, Void) || add!(x.grad, gy)
    Var(y, df, (x,))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)
