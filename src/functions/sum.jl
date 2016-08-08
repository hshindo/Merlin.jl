import Base.sum

"""
    sum(x, dim::Int)

Compute the sum along the given dimensions.
"""
function sum(x::Var, dim::Int)
    y = sum(x.data,dim)
    df(gy) = hasgrad(x) && broadcast!(.+, x.grad, x.grad, gy)
    Var(y, [x], sum, df)
end

sum(x::GraphNode, dim::Int) = GraphNode(sum, x, dim)
