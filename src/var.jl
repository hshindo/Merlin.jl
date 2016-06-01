type Var
  value
  grad
  f
  args::Vector{Var}
end

Var(value; grad=false) = Var(value, grad ? zeros(value) : nothing, nothing, Var[])
Var() = Var(nothing)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function forward0(f, args::Vector{Var})
  any(a -> a.value == nothing, args) && return Var(nothing, nothing, f, args)
  forward(f, args)
end
forward0(f, args::Var...) = forward0(f, [args...])
