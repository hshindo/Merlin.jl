export Var, zerograd, forward, gradient!

type Var
  value
  f
  args
  grad

  Var(value, f=nothing, args=[]) = new(value, f, args)
end

function zerograd{T}(value::T)
  v = Var(value)
  v.grad = zeros(value)
  v
end

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

function forward(f, args::Var...)
  any(a -> typeof(a.value) == Symbol, args) && return Var(Symbol(), f, args)
  f, y = f(map(a -> a.value, args)...)
  Var(y, f, args)
end

function forward{T<:Var}(f, args::Vector{T})
  any(a -> typeof(a.value) == Symbol, args) && return Var(Symbol(), f, args)
  f, y = f(map(a -> a.value, args))
  Var(y, f, args)
end

function backward!(y::Var, args::Tuple)
  xs = map(a -> a.value, args)
  gxs = map(a -> a.grad, args)
  backward!(y.f, xs..., gxs..., y.value, y.grad)
end

function backward!(y::Var, args::Vector)
  xs = map(a -> a.value, args)
  gxs = map(a -> a.grad, args)
  backward!(y.f, xs, gxs, y.value, y.grad)
end

function gradient!(top::Var)
  sorted = topsort(top)
  isdefined(top, :grad) || (top.grad = ones(top.value))
  for i = 1:length(sorted)-1 # excludes top
    v = sorted[i]
    isdefined(v, :grad) && continue
    isempty(v.args) && (v.grad = empty(typeof(v.value)))
    v.grad = zeros(v.value)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    isempty(v.args) || backward!(v, v.args)
  end
  sorted
end

"""
    flatten(pred, top::Var)

Flatten var graph
"""
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
