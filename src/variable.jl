type Variable
  value
  grad
  f
  args::Vector{Variable}
  backward!
end

Variable(value=nothing, grad=nothing) = Variable(value, grad, nothing, Variable[], nothing)
param(value) = Variable(value, zeros(value))

function forward(f::Functor, args::Vector{Variable})
  v = Variable(nothing, nothing, f, args, nothing)
  all(a -> a.value != nothing, args) && forward!(f, v)
  v
end
forward(f::Functor, arg::Variable) = forward(f, [arg])
forward(f::Functor, args::Variable...) = forward(f, [args...])

Base.getindex(v::Variable, key) = v.args[key]
Base.setindex!(v::Variable, value, key) = v.args[key] = value
Base.eltype(v::Variable) = eltype(v.value)

hasgrad(v::Variable) = v.grad != nothing

function gradient!(var::Variable)
  var.grad == nothing && (var.grad = ones(var.value))
  sorted = topsort(var)
  for v in sorted
    (v == var || hasgrad(v)) && continue
    v.backward == nothing || (v.grad = zeros(v.value))
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    v.backward == nothing || v.backward!()
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

function numerical_grad(f::Functor, xs::Vector, eps=1e-3)
  x1, x2 = copy(x), copy(x)
  gx = zeros(x)
  for i = 1:length(x)
    x1[i] += eps
    x2[i] -= eps
    y1 = f(Variable(x1))
    y2 = f(Variable(x2))
    gx[i] = sum(y1.value-y2.value) / (2*eps)
    x1[i] -= eps
    x2[i] += eps
  end
  gx
end

function gradient_check(f::Functor, args::Vector{Variable})
  out = f(args)
  gradient!(out)
  gx1 = out.grad
  gx2 = numerical_grad(f, x)
  gx1 - gx2
end
