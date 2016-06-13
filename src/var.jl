type Var
  value
  f
  args::Union{Vector{Var},Tuple{Vararg{Var}}}
  grad
end

Var(value, f=nothing, args=Var[], grad=nothing) = Var(value, f, args, grad)
param(value) = Var(value, nothing, Var[], zeros(value))

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

isparam(v::Var) = isempty(v.args) && v.grad != nothing

function forward(f::Functor, args::Var...)
  any(a -> typeof(a.value) == Symbol, args) && return Var(Symbol(), f, args)
  f, y = forward(f, map(a -> a.value, args)...)
  Var(y, f, args)
end

function forward(f::Functor, args::Vector{Var})
  any(a -> typeof(a.value) == Symbol, args) && return Var(Symbol(), f, args)
  f, y = forward(f, map(a -> a.value, args))
  Var(y, f, args)
end

function backward!(f::Functor, y::Var)
  if length(y.args) == 1
    x = y[1]
    x.grad == nothing && return
    backward!(f, x.value, x.grad, y.value, y.grad)
  else
    xs = map(a -> a.value, y.args)
    gxs = map(a -> a.grad, y.args)
    backward!(f, xs, gxs, y.value, y.grad)
  end
end

function gradient!(top::Var)
  sorted = topsort(top)
  top.grad == nothing && (top.grad = ones(top.value))
  for i = 1:length(sorted)-1 # excludes top
    v = sorted[i]
    v.grad == nothing || continue
    if isempty(v.args)
      v.grad = empty(typeof(v.value))
    else
      v.grad = zeros(v.value)
    end
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    v.f == nothing && continue
    backward!(v.f, v)
  end
  sorted
end

"""
    checkargs(expr)

Check arguments and decide eager or lazy evaluation.
"""
macro checkargs(f, args)
  quote
    if any(a -> typeof(a.value) == Symbol, $args)
      return Var(Symbol(), $f, $args)
    end
  end
end
