import Base.getindex

"""
    getindex(x::Var, inds...)

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = x[1:3]
y = x[2]
```
"""
@graph function getindex(x::Var, inds::Tuple)
    y = x.data[inds...]
    df(gy) = isconst(x) || (x.grad[inds...] .+= gy)
    Var(y, [x], getindex, df)
end
getindex(x::Var, inds...) = getindex(x, inds)
