type Var
  value
  f
  args::Vector{Var}
  grad
end

Var(value, f=nothing, args=Var[], grad=nothing) = Var(value, f, args, grad)
Var() = Var(nothing)
param(value) = Var(value, nothing, Var[], zeros(value))

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function forward(f, args::Vector{Var})
  any(a -> a.value == nothing, args) && return Var(nothing, nothing, f, args)
  f(args)
end
