export Var, zerograd, forward, gradient!

type Var{T}
  value::T
  f
  args
  grad::T

  Var(value, f, args) = new(value, f, args)
end

Var{T}(value::T, f=nothing, args=[]) = Var{T}(value, f, args)

function zerograd{T}(value::T)
  v = Var(value)
  v.grad = zeros(value)
  v
end

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

function init(f::Functor, args::Tuple)
  any(a -> typeof(a.value) == Symbol, args) && return Var(Symbol(), f, args)
  f, y = forward(f, args...)
  Var(y, f, args)
end

function forward(f::Functor, args::Vector{Var})
  any(a -> typeof(a.value) == Symbol, args) && return Var(Symbol(), f, args)
  f, y = forward(f, map(a -> a.value, args))
  Var(y, f, args)
end

function backward!(f::Functor, y::Var)
  if typeof(y.args) <: Tuple
    backward!(f, y.args..., y)
  else
    throw("error")
  end
end

function gradient!(top::Var)
  sorted = topsort(top)
  isdefined(top, :grad) || (top.grad = ones(top.value))
  for i = 1:length(sorted)-1 # excludes top
    v = sorted[i]
    isdefined(v, :grad) && continue
    isempty(v.args) && continue
    v.grad = zeros(v.value)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    isempty(v.args) && continue
    if typeof(v.args) <: Tuple
      backward!(v.f, v.args..., v)
    else
      throw("error")
    end
  end
  sorted
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
