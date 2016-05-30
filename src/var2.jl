export Var
export backward!, approx_grad, checkgrad

type Var
  val
  grad
  f
  args::Vector{Var}
  backward!
end

function Var(val; grad=nothing, f=nothing, args=Var[], (backward!)=nothing)
  Var(val, grad, f, args, backward!)
end
Var() = Var(nothing)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, val, key) = v.args[key] = val

hasgrad(v::Var) = v.grad != nothing

function forward0(f::Functor, args::Vector{Var})
  any(a -> a.val == nothing, args) && return Var(nothing, f=f, args=args)
  forward(f, args)
end
forward0(f::Functor, arg::Var) = forward0(f, [arg])
forward0{N}(f::Functor, args::NTuple{N,Var}) = forward0(f, [args...])

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
    v.backward! == nothing || v.backward!(v.grad)
  end
  sorted
end

"""
Compute numerical gradient.
"""
function approx_grad(f::Functor, args::Vector{Var})
  epsilon = 1e-3
  map(args) do x
    x = x.val
    gx = zeros(x)
    origx = copy(x)
    for k = 1:length(x)
      x[k] = origx[k] + epsilon
      y1 = f(args).val
      x[k] = origx[k] - epsilon
      y2 = f(args).val
      x[k] = origx[k]
      gx[k] = sum(y1 - y2) / 2epsilon
    end
    copy!(x, origx)
    gx
  end
end

"""
Check gradient.
"""
function checkgrad(f::Functor, args::Vector{Var})
  y = f(args)
  for x in args
    x.grad = zeros(x.val)
  end
  backward!(y)
  approx_gxs = approx_grad(f, args)
  for i = 1:length(args)
    gx1 = args[i].grad
    gx2 = approx_gxs[i]
    all(d -> abs(d) < 1e-4, gx1 - gx2) || return false
  end
  true
end
checkgrad(f::Functor, args::Var...) = checkgrad(f, [args...])
