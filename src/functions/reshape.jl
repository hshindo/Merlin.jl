import Base: reshape

doc"""
    reshape(x, dims::Tuple)

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
y = reshape(x, (2,5), [2,3])
```
"""
function reshape(x::Var, dims::Tuple)
    y = isvoid(x.data) ? nothing : reshape(x.data,dims)
    Var(y, (reshape,x))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)

function addgrad!(y::Var, ::typeof(reshape), x::Var)
    if !isvoid(x.grad)
        T = eltype(x)
        BLAS.axpy!(T(1), y.grad, x.grad)
    end
end
