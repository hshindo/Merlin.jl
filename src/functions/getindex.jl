import Base: getindex

doc"""
    getindex(x::Var, inds...)

```julia
x = Var(rand(Float32,10,5))
y = x[1:3]
y = x[2:2]
```
Note that `y = x[i]` throws an error since `y` is not a vector but a scholar.
Instead, use `y = x[i:i]`.
"""
getindex(x::Var, inds...) = Var(x.data[inds...], (getindex,x,inds))
getindex(x::Node, inds...) = Node(getindex, x, inds)

function getindex(x::Var, index::Vector{I}) where I<:Integer
    Var(x.data[index], (getindex,x,index))
end

function addgrad!(y::Var, ::typeof(getindex), x::Var, inds::Tuple)
    isvoid(x.grad) && return
    gx = view(x.grad, inds...)
    axpy!(eltype(y)(1), y.grad, gx)
end

function addgrad!(y::Var, ::typeof(getindex), x::Var, index::Vector{I}) where I<:Integer
    addgrad!(y, getindex, x, (index,))
end
