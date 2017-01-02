import Base.view

"""
    view(x::Var, inds...)
"""
function view(x::Var, inds::Tuple)
    isvoid(x.data) && return Var(nothing, view, (x,inds))
    y = view(x.data, inds...)
    function df(gy)
        isvoid(x.grad) && return
        x.grad[inds...] = x.grad[inds...] + gy
    end
    Var(y, df, (x,))
end
view(x::Var, inds...) = view(x, inds)
