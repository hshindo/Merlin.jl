doc"""
    reshape(x, dims::Int...)
    reshape(x, dims::Tuple)
"""
function Base.reshape(x::Var, dims::Tuple)
    Var(reshape(x.data,dims), reshape, (x,dims))
end
Base.reshape(x::Var, dims::Int...) = reshape(x, dims)
Base.reshape(x::Node, dims::Tuple) = Node(reshape, x, dims)
Base.reshape(x::Node, dims::Int...) = reshape(x, dims)

function addgrad!(y::Var, ::typeof(reshape), x::Var, dims::Tuple)
    isvoid(x.grad) && return
    T = eltype(x.data)
    BLAS.axpy!(T(1), y.grad, x.grad)
end
