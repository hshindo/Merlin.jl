import Base: +, -

"""
    +(x1::Var, x2::Var)
    +(a::Number, x::Var)
    +(x::Var, a::Number)

```julia
y = Var([1.,2.,3.]) + Var([4.,5.,6.])
y = 1.0 + Var([4.,5.,6.])
y = Var([1.,2.,3.]) + 4.0
```
"""
+(x1::Var, x2::Var) = axsum([1.0,1.0], [x1,x2])
+(a::Number, x::Var) = constant(a) + x
+(x::Var, a::Number) = x + constant(a)

"""
    -(x1::Var, x2::Var)
    -(a::Number, x::Var)
    -(a::Number, x::Var)
    -(x::Var)

See `+` for examples.
"""
-(x1::Var, x2::Var) = axsum([1.0,-1.0], [x1,x2])
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)
-(x::Var) = axsum([-1.0], [x])
