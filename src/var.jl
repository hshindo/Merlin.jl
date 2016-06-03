type Var
  value
  grad
  f
  args::Vector{Var}
end

Var() = Var(nothing, nothing, nothing, Var[])
param(value) = Var(value, zeros(value), nothing, Var[])

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function init(f, args::Vector{Var})
  any(a -> a.value == nothing, args) && return Var(nothing, nothing, f, args)
  f, y = forward(f, args)
  Var(y, nothing, f, args)
end
