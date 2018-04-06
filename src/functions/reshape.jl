import Base: reshape, vec

doc"""
    reshape(x, dims::Tuple)

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
y = reshape(x, 5, 10)
```
"""
reshape(x::Var, dims::Tuple) = Var(reshape(x.data,dims), (reshape,x))
reshape(x::Var, dims::Int...) = reshape(x, dims)
vec(x::Var) = reshape(x, length(x))

function addgrad!(y::Var, ::typeof(reshape), x::Var)
    isvoid(x.grad) && return
    T = eltype(x)
    BLAS.axpy!(T(1), y.grad, x.grad)
end
