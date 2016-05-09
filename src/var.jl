export Var
export backward!, approx_gradient, check_gradient

type Var{T}
  val::T
  grad::Nullable{T}
  f
  args::Vector{Var}
  backward!
end

Var{T}(val::T, grad::Nullable{T}) = Variable(val, grad, nothing, Variable[], nothing)
Var{T}(val::T, grad::T) = Variable(val, Nullable{T}(grad))
Var{T}(val::T) = Variable(val, Nullable{T}())
Var() = Var(nothing)
Var{T}(val::T, f, args, backward!) = Variable(val, Nullable{T}(), f, args, backward!)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, val, key) = v.args[key] = val

function topsort(var::Var)
  sorted = Var[]
  dict = ObjectIdDict()
  function visit(v::Var)
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

function backward!(var::Var)
  hasgrad(var) || (var.grad = ones(var.val))
  sorted = topsort(var)
  for v in sorted
    (v == var || hasgrad(v) || v.backward! == nothing) && continue
    v.grad = zeros(v.val)
  end
  for i = length(sorted):-1:1
    v = sorted[i]
    v.backward! == nothing || v.backward!()
  end
  sorted
end

"""
Computes numerical gradient.
"""
function approx_gradient{N}(f::Functor, xs::NTuple{N,Data})
  epsilon = 1e-4
  map(xs) do x
    gx = zeros(x)
    orig = copy(x)
    for k = 1:length(x)
      x[k] = orig[k] + epsilon
      y1 = f(xs).val
      x[k] = orig[k] - epsilon
      y2 = f(xs).val
      x[k] = orig[k]
      gx[k] = sum(y1 - y2) / 2epsilon
    end
    copy!(x, orig)
    gx
  end
end
approx_gradient(f::Functor, x::Data) = approx_gradient(f, (x,))[1]

function gradient2{N}(f::Functor, xs::NTuple{N,Data})
  inputs = map(Variable, xs)
  out = f(inputs)
  gradient!(out)
  map(v -> v.grad, inputs)
end
gradient2(f::Functor, xs::Data...) = gradient2(f, xs)

"""
Check gradient.
"""
function check_gradient{N}(f::Functor, xs::NTuple{N,Data})
  inputs = map(Variable, xs)
  out = f(inputs)
  gradient!(out)
  approx_gxs = approx_gradient(f, xs)
  for i = 1:length(xs)
    gx1 = inputs[i].grad
    gx2 = approx_gxs[i]
    all(d -> abs(d) < 1e-4, gx1 - gx2) || return false
  end
  true
end
#check_gradient(f::Functor, x::Data) = check_gradient(f, (x,))
check_gradient(f::Functor, xs::Data...) = check_gradient(f, xs)
