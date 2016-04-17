type Variable
  value
  grad
  f
  args::Vector{Variable}
  backward!
end

Variable(value, grad) = Variable(value, grad, nothing, Variable[], nothing)
Variable(value) = Variable(value, zeros(value))
Variable(value) = Variable(nothing, nothing)

function forward(f::Functor, args::Vector{Variable})
  v = Variable(nothing, nothing, f, args, nothing)
  for a in args
    a.value == nothing && return v
  end
  forward!(f, v)
  v
end
forward(f::Functor, arg::Variable) = forward(f, [arg])
forward(f::Functor, args::Variable...) = forward(f, [args...])
forward{T<:Any}(f::Functor, args::Vector{T}) = forward(f, map(Variable, args))
forward(f::Functor, arg::Any) = forward(f, [Variable(arg)])
function forward(f::Functor, args...)
  vars = Variable[]
  for a in args
    v = typeof(a) == Variable ? a : Variable(a, nothing)
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
Computes numerical gradient for gradient check.
"""
function approx_gradient(f::Functor, x, eps=1e-3)
  x1, x2 = copy(x), copy(x)
  gx = zeros(x)
  for i = 1:length(x)
    x1[i] += eps
    x2[i] -= eps
    out1 = f(Variable(x1,zeros(x1)))
    out2 = f(Variable(x2,zeros(x2)))
    gx[i] = sum(out1.value - out2.value) / 2eps
    x1[i] -= eps
    x2[i] += eps
  end
  gx
end

function test_gradient(f::Functor, x)
  arg = Variable(x, zeros(x))
  out = f(arg)
  gradient!(out)
  arg.grad
end
