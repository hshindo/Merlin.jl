import Base.view

"""
    view(x::Var, inds...)
"""
view(x::Var, inds::Tuple) = forward(view, x, inds)
view(x::Var, inds...) = view(x, inds)

function forward(::typeof(view), x::Array, inds::Tuple)
    y = view(x, inds...)
    backward!(gy, gx) = isvoid(gx) || (gx[inds...] += gy)
    y, backward!
end
