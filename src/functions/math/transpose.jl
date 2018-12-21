import Base.transpose

"""
    transpose(x)
"""
function transpose(x::Var)
    y = permutedims(x.data)
    Var(y, ∇transpose!, (x,))
end

function ∇transpose!(y::Var, x::Var)
    isnothing(x.grad) && return
    addto!(x.grad, permutedims(y.grad))
end
