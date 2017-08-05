import Base.getindex

"""
    getindex(x::Var, inds...)

```julia
x = Var(rand(Float32,10,5))
y = x[1:3]
y = x[2:2]
```
Note that `y = x[i]` throws an error since `y` is not a vector but a scholar.
Instead, use `y = x[i:i]`.
"""
function getindex(x::Var, inds::Tuple)
    #v = view(x.data, f.inds...)
    #data = islinear(v) ? unsafe_wrap(Array,pointer(v),size(v)) : x.data[f.inds...]

    data = x.data[inds...]
    Var(data, x.batchdims, getindex, (x,inds))
end
getindex(x::Var, inds::Union{Int,Range,Colon}...) = getindex(x, inds)

getindex(x::Node, inds::Tuple) = Node(getindex, x, inds)
getindex(x::Node, inds::Union{Int,Range,Colon}...) = getindex(x, inds)

function addgrad!(y::Var, ::typeof(getindex), x::Var, inds::Tuple)
    isvoid(x.grad) && return
    gx = view(x.grad, inds...)
    broadcast!(+, gx, gx, y.grad)
end
