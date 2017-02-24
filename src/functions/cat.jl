import Base.cat

"""
    cat(dim::Int, xs::Var...)
    cat(dim::Int, xs::Vector{Var})

Concatenate arrays over the given dimension.

```julia
x1 = Var(rand(Float32,4,3))
x2 = Var(rand(Float32,4,5))
y = cat(2, x1, x2)
y = cat(2, Var[x1,x2])
```
"""
function cat{T<:UniArray}(dim::Int, xs::Vector{T})
    cumdim = 0
    for x in xs
        cumdim += size(x, dim)
    end
    outsize = [size(xs[1])...]
    while length(outsize) < dim
        push!(outsize, 1)
    end
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
    backward!(gy, gxs) = ∇cat!(gy, dim, gxs)
    Var(y, cat, (dim,xs), df)
end
cat(dim::Int, xs::Var...) = cat(dim, Var[xs...])

function ∇cat!{T<:UniArray}(gy, dim::Int, xs::Vector{T}, gxs::Vector)
    range = Any[1:size(gy,i) for i=1:ndims(gy)]
    offset = 1
    for i = 1:length(xs)
        x, gx = xs[i], gxs[i]
        s = size(x, dim)
        if isvoid(gx)
            offset += s
        else
            if dim > ndims(gx)
                range[dim] = offset
            else
                range[dim] = offset:(offset+s-1)
            end
            broadcast!(+, gx, gx, view(gy,range...))
            offset += s
        end
    end
end
