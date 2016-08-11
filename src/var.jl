export Var, Param

type Var <: AbstractNode
    data
    args::Vector{Var}
    f
    df
    grad
end

Var(data, args, f, df=nothing, grad=nothing) = Var(data, args, df, grad)
Var(data) = Var(data, Var[], nothing)

Param(data) = Var(data, Var[], nothing, nothing, zeros(data))

hasgrad(v::Var) = v.grad != nothing

Base.size(v::Var) = size(v.data)
Base.size(v::Var, dim::Int) = size(v.data, dim)
Base.length(v::Var) = length(v.data)
