import Base.view

"""
    view(x::Var, inds...)
"""
view(x::Var, inds::Tuple) = forward(view, x, inds)
view(x::Var, inds...) = view(x, inds)

function forward(::typeof(view), x::Array, inds::Tuple)
    y = view(x, inds...)
    function backward!(gy, gx, inds)
        isvoid(gx) && return
        gx[inds...] += gy
    end
    y, backward!
end
