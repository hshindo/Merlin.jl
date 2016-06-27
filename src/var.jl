export Var, param, gradient!

type Var
  value
  f
  args
  grad
end

function Base.show(io::IO, v::Var)
  print(io, "Merlin.Var($(typeof(v.value)),$(v.f),$(v.args),$(typeof(v.grad)))")
end

Var(value, f=nothing, args=[]) = Var(value, f, args, nothing)
param(value) = Var(value, nothing, [], zeros(value))

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function settype!{T}(::Type{T}, vars::Vector{Var})
  for v in vars
    typeof(v.value) <: T && continue
    v.value = T(v.value)
  end
end

function gradient!(top::Var)
  sorted = topsort(top)
  hasgrad(top) || (top.grad = ones(top.value))
  for i = 1:length(sorted)-1 # excludes top
    v = sorted[i]
    hasgrad(v) && continue
    isempty(v.args) || (v.grad = zeros(v.value))
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    v.f == nothing || v.f(v.grad)
  end
  sorted
end

function flatten(pred::Function, top::Var)
  args = Var[]
  for v in top.args
    vv = flatten(v)
    pred(vv) && append!(args, vv.args)
  end
  Var(v.value, v.f, args, v.grad)
end

"""
    checkargs(expr)

Check arguments and decide eager or lazy evaluation..
"""
macro checkargs(f, args)
  quote
    if any(a -> typeof(a.value) == Symbol, $args)
      return Var(Symbol(), $f, $args)
    end
  end
end
