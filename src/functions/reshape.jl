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
    configure!(x)
    Var(reshape(x.data,dims), (reshape,x))
end
reshape(x::Var, dims::Int...) = reshape(x, dims)

vec(x::Var) = reshape(x, length(x))

function addgrad!(y::Var, ::typeof(reshape), x::Var)
    isvoid(x.grad) && return
    addto!(x.grad, y.grad)
end

doc"""
    dropdims(x, dims::Tuple)

Remove the dimensions of `x` specified by dims.
"""
function dropdims(x::Var, dims::Tuple)
    configure!(x)
    y = dropdims(x.data, dims)
    Var(y, (dropdims,x,dims))
end
dropdims(x::Var, dims::Int...) = dropdims(x, dims)
function dropdims(x::Var)
    dims = Int[]
    for d in 1:ndims(x)
        size(x,d) == 1 && push!(dims,d)
    end
    dropdims(x, dims...)
end

function addgrad!(y::Var, ::typeof(dropdims), x::Var, dims)
    isvoid(x.grad) && return
    addto!(x.grad, y.grad)
end
