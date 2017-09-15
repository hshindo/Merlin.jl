import Base.reshape

doc"""
    reshape(x, dims::Int...)
    reshape(x, dims::Tuple)
"""
function reshape(x::Var, dims::Tuple)
    throw("Not implemented yet.")
    Var(reshape(x.data,dims), x.batchdims, reshape, (x,dims))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)

reshape(x::Node, dims::Tuple; name="reshape") = Node(reshape, x, dims, name=name)
reshape(x::Node, dims::Int...) = reshape(x, dims)

function addgrad!(y::Var, ::typeof(reshape), x::Var, dims::Tuple)
    isvoid(x.grad) && return
    T = eltype(x.data)
    BLAS.axpy!(T(1), y.grad, x.grad)
end
