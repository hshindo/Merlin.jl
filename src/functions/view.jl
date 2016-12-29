import Base.view

"""
    view(x::Var, inds...)
"""
function view(x::Var, inds::Tuple)
    y = view(x.data, inds...)
    function df(gy)
        isvoid(x.grad) && return
        x.grad[inds...] = x.grad[inds...] + gy
    end
    Var(y, df, (x,))
end
view(x::Var, inds...) = view(x, inds)
view(x::Var{Void}, inds::Tuple) = Var(Void(), view, (x,inds))
