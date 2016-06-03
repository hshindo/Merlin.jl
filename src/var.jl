type Var
  value
  grad
  f
  args::Vector{Var}
end

Var(value) = Var(value, nothing, nothing, Var[])
Var() = Var(nothing)
param(value) = Var(value, zeros(value), nothing, Var[])

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function forward(f, args::Vector{Var})
  any(a -> a.value == nothing, args) && return Var(nothing, nothing, f, args)
  f, y = f(map(a -> a.value, args)...)
  Var(y, nothing, f, args)
end
forward(f, args::Var...) = forward(f, [args...])
