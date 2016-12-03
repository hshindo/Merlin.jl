export concat

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays along the given dimension.

```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = concat(2, x1, x2)
```
"""
function concat(dim::Int, xs::Vector{Var})
    cumdim = 0
    for x in xs
        cumdim += size(x.data, dim)
    end
    outsize = [size(xs[1].data)...]
    outsize[dim] = cumdim
    y = similar(xs[1].data, outsize...)
    range = map(s -> 1:s, outsize)
    offset = 1
    for x in xs
        s = size(x.data, dim)
        range[dim] = offset:(offset+s-1)
        y[range...] = x.data
        offset += s
    end
    df(gy) = ∇concat!(gy, dim, xs)
    Var(y, concat, xs, df)
end

function concat(dim::Int, xs::Var...)
    for x in xs
        x.data == nothing && return Var(nothing, concat, (dim,xs...))
    end
    concat(dim, [xs...])
end

function ∇concat!(gy::Array, dim::Int, xs::Vector{Var})
    range = [1:size(gy,i) for i=1:ndims(gy)]
    offset = 1
    for x in xs
        isconst(x) && continue
        s = size(x.data, dim)
        range[dim] = offset:(offset+s-1)
        broadcast!(+, x.grad, x.grad, view(gy,range...))
        offset += s
    end
end
