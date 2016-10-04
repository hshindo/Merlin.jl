import Base.sum

"""
    sum(x, dim::Int)

Compute the sum along the given dimensions.
"""
@graph function sum(x::Var, dim::Int)
    y = sum(x.data,dim)
    df(gy) = isconst(x) || broadcast!(.+, x.grad, x.grad, gy)
    Var(y, [x], sum, df)
end
