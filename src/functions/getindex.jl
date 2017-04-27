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
getindex(x::Var, inds::Tuple) = GetIndex(inds)(x)
getindex(x::Var, inds...) = getindex(x, inds)

#islinear{T,N,P,I,L}(x::SubArray{T,N,P,I,L}) = L

type GetIndex
    inds::Tuple
end

function (f::GetIndex)(x::Var)
    #v = view(x.data, f.inds...)
    #data = islinear(v) ? unsafe_wrap(Array,pointer(v),size(v)) : x.data[f.inds...]
    data = x.data[f.inds...]
    y = Var(data, f, (x,))
    y.df! = function df!()
        isvoid(x.grad) && return
        gx = view(x.grad, f.inds...)
        broadcast!(+, gx, gx, y.grad)
    end
    y
end
