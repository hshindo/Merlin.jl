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
@forward getindex(x::Var, inds::Tuple)

getindex(x::Var, inds...) = getindex(x, inds)

function forward{T}(::typeof(getindex), x::Array{T}, inds::Tuple)
    y = x[inds...]
    function backward!(gy, gx)
        isvoid(gx) && return
        #BLAS.axpy!(T(1), gx[inds...])
        gx = view(gx, inds...)
        broadcast!(+, gx, gx, gy)
    end
    y, backward!
end
