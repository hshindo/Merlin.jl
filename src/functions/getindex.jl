import Base.getindex

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
    @assert ndims(x) == length(inds)
    subxs = unsafe_split(x.data, x.batchdims)
    if !all(subx -> validindex(size(subx,ndims(subx)),inds[end]), subxs)
        issorted(x.batchdims,rev=true) || throw("x.batchdims must be sorted in descending order: $(x.batchdims).")
    end

    ys = []
    for s in subxs
        validindex(size(s,ndims(x)), inds[end]) || break
        push!(ys, s[inds...])
    end
    y = cat(ndims(x), ys...)
    batchdims = map(y -> size(y,ndims(x)), ys)
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

validindex(dim::Int, ind::Int) = dim >= ind
validindex(dim::Int, ind::Range) = dim >= last(ind)
validindex(dim::Int, ind::Colon) = true
