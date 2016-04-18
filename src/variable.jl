export Variable
export hasgrad, gradient!, gradient, topsort, approx_gradient

type Variable
  value
  grad
  f
  args::Tuple{Vararg{Variable}}
  backward!
end

Variable(value, grad) = Variable(value, grad, nothing, (), nothing)
Variable(value) = Variable(value, zeros(value))
Variable() = Variable(nothing, nothing)

function forward(f::Functor, args::Vector{Variable})
  v = Variable(nothing, nothing, f, args, nothing)
  for a in args
    a.value == nothing && return v
  end
  forward!(f, v)
  v
end
forward(f::Functor, arg::Variable) = forward(f, [arg])
forward(f::Functor, args...) = forward(f, Any[args...])
function forward(f::Functor, args::Vector{Any})
  vars = Variable[]
  for a in args
    v = typeof(a) == Variable ? a : Variable(a, nothing)
    push!(vars, v)
  end
  forward(f, vars)
end

Base.getindex(v::Variable, key) = v.args[key]
Base.setindex!(v::Variable, value, key) = v.args[key] = value
Base.eltype(v::Variable) = eltype(v.value)

hasgrad(v::Variable) = v.grad != nothing

function gradient!(var::Variable)
  hasgrad(var) || (var.grad = ones(var.value))
  sorted = topsort(var)
  for v in sorted
    (v == var || hasgrad(v) || v.backward! == nothing) && continue
    v.grad = zeros(v.value)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    v.backward! == nothing || v.backward!()
  end
end

function topsort(var::Variable)
  sorted = Variable[]
  dict = ObjectIdDict()
  function visit(v::Variable)
    if !haskey(dict, v)
      dict[v] = v
      for a in v.args
        visit(a)
      end
      push!(sorted, v)
    end
  end
  visit(var)
  sorted
end

"""
Computes numerical gradient.
"""
function approx_gradient(f::Functor, xs::Vector{Any}, eps=1e-3)
  inputs = map(xs) do x
    grad = applicable(zeros, x) ? zeros(x) : nothing
    Variable(x, grad)
  end
  for v in inputs
    hasgrad(v) || continue
    for k = 1:length(x)
      v.value[k] += eps
      y1 = f(inputs).value
      v.value[k] -= 2eps
      y2 = f(inputs).value
      v.value[k] += eps
      v.grad[k] = sum(y1 - y2) / 2eps
    end
  end
  map(v -> v.grad, inputs)
end
approx_gradient(f::Functor, xs::Tuple, eps=1e-3) = approx_gradient(f, Any[xs...], eps)
approx_gradient(f::Functor, x, eps=1e-3) = approx_gradient(f, Any[x], eps)[1]

function gradient(f::Functor, xs::Vector{Any})
  inputs = map(xs) do x
    grad = applicable(zeros, x) ? zeros(x) : nothing
    Variable(x, grad)
  end
  out = f(inputs)
  gradient!(out)
  map(v -> v.grad, inputs)
end
gradient(f::Functor, x)
