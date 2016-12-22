import Base.getindex

"""
    getindex(x::Var, inds...)

```julia
x = Var(rand(Float32,10,5))
y = x[1:3]
y = x[2:2]
```
"""
getindex(x::Var{Void}, inds...) = Var(Void(), getindex, (x,inds))

function getindex(x::Var, inds::Tuple)
    y = x.data[inds...]
    function df(gy)
        isvoid(x.grad) && return
        gx =view(x.grad, inds...)
        broadcast!(+, gx, gx, gy)
    end
    Var(y, df, (x,))
end
getindex(x::Var, inds...) = getindex(x, inds)
