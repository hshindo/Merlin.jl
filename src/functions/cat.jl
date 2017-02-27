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
cat(dim::Int, xs::Vector{Var}) = forward0(dim, xs)
cat(dim::Int, xs::Var...) = cat(dim, Var[xs...])

function forward{T<:UniArray}(::typeof(cat), dim::Int, xs::Vector{T})
    y = cat(dim, xs...)
    backward!(gy, gxs) = ∇cat!(gy, dim, xs, gxs)
    y, backward!
end

function ∇cat!{T}(gy::UniArray{T}, dim::Int, xs::Vector, gxs::Vector)
    range = Any[1:size(gy,i) for i=1:ndims(gy)]
    offset = 1
    for i = 1:length(xs)
        x, gx = xs[i], gxs[i]
        s = size(x, dim)
        if isvoid(gx)
            offset += s
        else
            range[dim] = offset:(offset+s-1)
            #if dim > ndims(gx)
            #    range[dim] = offset
            BLAS.axpy!(T(1), gy[range...], gx)
            offset += s
        end
    end
end
