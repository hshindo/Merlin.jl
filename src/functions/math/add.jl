import Base: +, -

"""
    +(x1::Var, x2::Var)
    +(a::Number, x::Var)
    +(x::Var, a::Number)
"""
+(x1::Var, x2::Var) = axsum([1.0,1.0], [x1,x2])
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

"""
    -(x1::Var, x2::Var)
    -(a::Number, x::Var)
    -(a::Number, x::Var)
    -(x::Var)
"""
-(x1::Var, x2::Var) = axsum([1.0,-1.0], [x1,x2])
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)
-(x::Var) = axsum([-1.0], [x])
