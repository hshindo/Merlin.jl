import Base.reshape

type Reshape <: Var
    data
    grad
    tails::Vector
    dims::Tuple
end

"""
reshape(x::Var, dims::Int...)

Reshape an array according to the given dimensions.

### 👉 Example
```julia
x = Data(rand(Float32,10,5,3))
y = reshape(x, 5, 3, 5, 2)
```
"""
function reshape(x::Var, dims::Tuple)
    y = hasdata(x) ? copy(reshape(x.data,dims)) : nothing
    Reshape(y, nothing, [x], dims)
end
reshape(x::Var, dims::Int...) = reshape(x, dims)
@compat (f::Reshape)(x::Var) = reshape(x, f.dims)

function backward!(y::Reshape)
    hasgrad(y[1]) || return
    BLAS.axpy!(eltype(y.grad)(1), y.grad, y[1].grad)
end
