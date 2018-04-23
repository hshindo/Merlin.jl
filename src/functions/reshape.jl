import Base: reshape, squeeze, vec

doc"""
    reshape(x, dims::Tuple)

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
y = reshape(x, 5, 10)
```
"""
function reshape(x::Var, dims::Tuple)
    configure!(x)
    Var(reshape(x.data,dims), (reshape,x))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)

vec(x::Var) = reshape(x, length(x))

function addgrad!(y::Var, ::typeof(reshape), x::Var)
    isvoid(x.grad) && return
    T = eltype(x)
    BLAS.axpy!(T(1), y.grad, x.grad)
end

doc"""
    squeeze(x, dims::Tuple)

Remove the dimensions of `x` specified by dims.
"""
function squeeze(x::Var, dims::Tuple)
    configure!(x)
    y = squeeze(x.data, dims)
    Var(y, (squeeze,x,dims))
end
squeeze(x::Var, dims::Int...) = squeeze(x, dims)
function Base.squeeze(x::Var)
    dims = Int[]
    for d in 1:ndims(x)
        size(x,d) == 1 && push!(dims,d)
    end
    squeeze(x, dims...)
end

function addgrad!(y::Var, ::typeof(squeeze), x::Var, dims)
    isvoid(x.grad) && return
    T = eltype(x)
    BLAS.axpy!(T(1), y.grad, x.grad)
end
