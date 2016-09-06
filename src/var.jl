export Var, constant, isconstant

type Var
    data
    grad
    args::Vector{Var}
    f
    df
end

Var(data, args, f, df) = Var(data, nothing, args, f, df)
Var(data, grad) = Var(data, grad, Var[], nothing, nothing)
Var(data) = Var(data, zeros(data))

hasgrad(v::Var) = v.grad != nothing

constant(data) = Var(data)
isconstant(v::Var) = v.grad == nothing

h5convert(x::Var) = h5dict(Var, "data"=>h5convert(x.data))
h5load!(::Type{Var}, data::Dict) = Var(data["data"])
