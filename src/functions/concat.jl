import Base.cat

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays over the given dimension.

```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = concat(2, x1, x2)
y = concat(2, Var[x1,x2])
```
"""
cat(dim::Int, xs::Var...) = forward(cat, dim, xs...)

function forward(::typeof(cat), dim::Int, xs::Array...)
    cumdim = 0
    for x in xs
        cumdim += size(x, dim)
    end
    outsize = [size(xs[1])...]
    outsize[dim] = cumdim
    y = similar(xs[1], outsize...)
    range = map(s -> 1:s, outsize)
    offset = 1
    for x in xs
        s = size(x, dim)
        range[dim] = offset:(offset+s-1)
        y[range...] = x
        offset += s
    end
    backward!(gy, dim, gxs...) = ∇concat!(gy, dim, gxs...)
    y, backward!
end

function ∇concat!(gy, dim::Int, gxs...)
    range = [1:size(gy,i) for i=1:ndims(gy)]
    offset = 1
    for gx in gxs
        isvoid(gx) && continue
        s = size(gx, dim)
        range[dim] = offset:(offset+s-1)
        broadcast!(+, gx, gx, view(gy,range...))
        offset += s
    end
end
