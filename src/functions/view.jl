import Base.view

"""
    view(x::Var, inds...)
"""
@graph function view(x::Var, inds::Tuple{Vararg{Int}})
    y = view(x.data, inds...)
    df(gy) = isconst(x) || (x.grad[inds...] += gy)
    Var(y, [x], view, df)
end
view(x::Var, inds::Int...) = view(x, inds)
