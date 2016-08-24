export Var, zerograd!

type Var
    data
    grad
    args::Vector{Var}
    f
    df
end

Var(data, args, f, df) = Var(data, nothing, args, f, df)
Var(data, grad=nothing) = Var(data, grad, Var[], nothing, nothing)

hasgrad(v::Var) = v.grad != nothing

function zerograd!(v::Var)
    T = eltype(v.data)
    if typeof(v.data) <: UniArray
        hasgrad(v) ? fill!(v.grad, T(0)) : (v.grad = zeros(v.data))
    elseif typeof(v.data) <: Number
        v.grad = T(0)
    end
    v
end
