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
function getindex(x::Var, inds::Tuple)
    ys = []
    for xx in unsafe_split(x.data,x.batchdims)
        push!(ys, xx[inds...])
    end
    y = cat(ndims(ys[1]), ys...)
    batchdims = map(y -> size(y)[end], ys)
    Var(y, batchdims, getindex, (x,inds))
end
getindex(x::Var, inds::Union{Int,Range,Colon}...) = getindex(x, inds)

getindex(x::Node, inds::Tuple; name="") = Node(getindex, (x,inds), name)
getindex(x::Node, inds::Union{Int,Range,Colon}...) = getindex(x, inds)

function addgrad!(y::Var, ::typeof(getindex), x::Var, inds::Tuple)
    if !isvoid(x.grad)
        gxs = unsafe_split(x.grad, x.batchdims)
        gys = unsafe_split(y.grad, y.batchdims)
        for (gx,gy) in zip(gxs,gys)
            g = view(gx, inds...)
            broadcast!(+, g, g, gy)
        end
    end
end
