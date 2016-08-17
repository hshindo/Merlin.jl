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
function getindex(x::Var, inds...)
    y = x.data[inds...]
    df(gy) = hasgrad(x) && (x.grad[inds...] .+= gy)
    Var(y, [x], getindex, df)
end
