import Base: reshape, dropdims, vec

"""
    reshape(x, dims::Tuple)

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
y = reshape(x, 5, 10)
```
"""
function reshape(x::Var, dims::Tuple)
    isnothing(x.data) && return Var(nothing,reshape,(x,dims))
    configure!(x)
    Var(reshape(x.data,dims), ∇reshape!, (x,))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)

vec(x::Var) = reshape(x, length(x))

function ∇reshape!(y::Var, x::Var)
    isnothing(x.grad) && return
    addto!(x.grad, y.grad)
end

doc"""
    dropdims(x, dims::Tuple)

Remove the dimensions of `x` specified by dims.
"""
function dropdims(x::Var, dims::Tuple)
    isnothing(x.data) && return Var(nothing,dropdims,(x,dims))
    configure!(x)
    ydata = dropdims(x.data, dims=dims)
    Var(ydata, ∇dropdims!, (x,))
end
dropdims(x::Var, dims::Int...) = dropdims(x, dims)
function dropdims(x::Var)
    dims = ()
    for i = 1:ndims(x)
        if size(x,i) == 1
            dims = tuple(dims..., i)
        end
    end
    dropdims(x, dims)
end

function ∇dropdims!(y::Var, x::Var)
    isnothing(x.grad) && return
    addto!(x.grad, y.grad)
end
