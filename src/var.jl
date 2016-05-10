export Var
export backward!, approx_gradient, check_gradient

type Var
  val
  grad
  f
  args
  backward!
end

Var(val, grad) = Var(val, grad, nothing, Var[], nothing)
Var(val) = Var(val, nothing)
Var() = Var(nothing, nothing)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, val, key) = v.args[key] = val
Base.size(v::Var) = size(v.val)
Base.length(v::Var) = length(v.val)

hasgrad(v::Var) = v.grad != nothing

function forward(f::Functor, x::Var)
  forward(f, x)
end

function forward(f::Functor, xs::Tuple{Vararg{Var}})
  forward(f, Var[xs...])
end

function forward(f::Functor, xs::Vector{Var})
  forward(f, xs)
end

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
    v == var && continue
    hasgrad(v) || continue
    v.backward! == nothing && continue
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
function approx_gradient(f::Functor, xs::Vector{Var})
  epsilon = 1e-4
  map(xs) do x
    x = x.val
    gx = zeros(x)
    origx = copy(x)
    for k = 1:length(x)
      x[k] = origx[k] + epsilon
      y1 = f(xs).val
      x[k] = origx[k] - epsilon
      y2 = f(xs).val
      x[k] = origx[k]
      gx[k] = sum(y1 - y2) / 2epsilon
    end
    copy!(x, origx)
    gx
  end
end
approx_gradient(f::Functor, xs::Var...) = approx_gradient(f, Var[xs...])

"""
Check gradient.
"""
function check_gradient(f::Functor, xs::Vector{Var})
  for x in xs
    x.grad = zeros(x.val)
  end
  backward!(y)
  approx_gxs = approx_gradient(f, xs)
  for i = 1:length(xs)
    gx1 = xs[i].grad
    gx2 = approx_gxs[i]
    all(d -> abs(d) < 1e-4, gx1 - gx2) || return false
  end
  true
end
check_gradient(f::Functor, xs::Var...) = check_gradient(f, Var[xs...])
