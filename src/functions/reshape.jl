import Base.reshape

doc"""
    reshape(x, dims::Tuple, batchdims::Vector{Int})

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
y = reshape(x, (2,5), [2,3])
```
"""
function reshape(x::Var, dims::Tuple, batchdims::Vector{Int})
    y = reshape(x.data, dims)
    sum(batchdims) == size(y,ndims(y)) || throw("Invalid batchdims: $(size(y)) and $batchdims")
    Var(y, batchdims, reshape, (x,))
end

reshape(x::Node, dims::Tuple, batchdims::Vector{Int}; name="") = Node(reshape, (x,dims), name)

function addgrad!(y::Var, ::typeof(reshape), x::Var)
    if !isvoid(x.grad)
        T = eltype(x)
        BLAS.axpy!(T(1), y.grad, x.grad)
    end
end
