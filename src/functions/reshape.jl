import Base.reshape

"""
    reshape(x::Var, dims::Int...)
"""
function reshape(x::Var, dims::Tuple)
    Var(reshape(x.data,dims), x.batchdims, reshape, (x,dims))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)
reshape(x::Node, dims::Tuple) = Node(reshape, x, dims)
reshape(x::Node, dims::Int...) = reshape(x, dims)

function addgrad!(y::Var, ::typeof(reshape), x::Var, dims::Tuple)
    isvoid(x.grad) && return
    T = eltype(x.data)
    BLAS.axpy!(T(1), y.grad, x.grad)
end
