function view(A::Var, inds...)
    hasdata(A) || return View(nothing, nothing, [A], inds)
    y = view(A.data, inds...)
    View(y, nothing, [A], inds)
end

type View <: Var
    data
    grad
    tails::Vector
    inds
end

function backward!(v::View)
    error("Not implemented yet.")
end
