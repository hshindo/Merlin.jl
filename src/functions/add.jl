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
+(x1::GraphNode, x2::Var) = GraphNode(+, x1, x2)
+(x1::Var, x2::GraphNode) = GraphNode(+, x1, x2)
+(x1::GraphNode, x2::GraphNode) = GraphNode(+, x1, x2)
+(a::Number, x::Union{GraphNode,Var}) = constant(a) + x
+(x::Union{Var,GraphNode}, a::Number) = x + constant(a)

"""
    -(x1::Var, x2::Var)
    -(a::Number, x::Var)
    -(a::Number, x::Var)
    -(x::Var)

See `+` for examples.
"""
-(x1::Var, x2::Var) = axsum([1.0,-1.0], [x1,x2])
-(x1::GraphNode, x2::Var) = GraphNode(-, x1, x2)
-(x1::Var, x2::GraphNode) = GraphNode(-, x1, x2)
-(x1::GraphNode, x2::GraphNode) = GraphNode(-, x1, x2)
-(a::Number, x::Union{Var,GraphNode}) = constant(a) - x
-(x::Union{Var,GraphNode}, a::Number) = x - constant(a)
-(x::Var) = axsum([-1.0], [x])
-(x::GraphNode) = GraphNode(-, x)
