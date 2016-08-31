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
    v.grad = zeros(v.data)
    v
end

h5convert(x::Var) = h5dict(Var, "data"=>h5convert(x.data))
h5load!(::Type{Var}, data::Dict) = Var(data["data"])
