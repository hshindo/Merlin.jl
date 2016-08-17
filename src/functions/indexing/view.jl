import Base.view

function view(x::Var, inds...)
    y = view(x.data, inds...)
    df(gy) = x.grad[inds...] += gy
    Var(y, [x], view, df)
end
