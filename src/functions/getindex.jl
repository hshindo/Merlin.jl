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
getindex(x::Var, inds::Tuple) = Var(x.data[inds...], getindex, (x,inds))
getindex(x::Var, inds...) = getindex(x, inds)
getindex(x::Node, inds::Tuple; name="") = Node(getindex, (x,inds), name)

function addgrad!(y::Var, ::typeof(getindex), x::Var, inds::Tuple)
    if !isvoid(x.grad)
        gx = view(x.grad, inds...)
        broadcast!(+, gx, gx, y.grad)
    end
end
