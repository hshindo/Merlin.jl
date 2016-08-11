"""
    getindex(x::Var, inds...)
"""
function Base.getindex(x::Var, inds::Union{Range,Colon,Int}...)
    y = x.data[inds...]
    df(gy) = hasgrad(x) && (x.grad[inds...] += gy) # TODO: more efficient in-place operation
    Var(y, [x], getindex, df)
end
