export Linear, linear

type Linear <: Var
  data
  grad
  tails::Vector
end

function Linear(T::Type, indim::Int, outdim::Int)
  r = T(sqrt(6 / (indim+outdim)))
  w = rand(-r, r, outdim, indim)
  b = fill(T(0), outdim, 1)
  Linear(nothing, nothing, [Param(w),Param(b),Data()])
end

@compat (l::Linear)(x::Var) = linear(l[1], l[2], x)

forward(l::Linear, w::Var, b::Var, x::Var) = linear(w, b, x)
forward(l::Linear, xs::Vector) = linear(xs[1], xs[2], xs[3])

function linear(w::Var, b::Var, x::Var)
  !hasdata(w) || !hasdata(b) || !hasdata(x) && return Linear(nothing, nothing, [w,b,x])
  y = w.data * x.data
  broadcast!(.+, y, y, b.data)
  Linear(y, nothing, [w,b,x])
end

function backward!(v::Linear)
  w, b, x = v[1], v[2], v[3]
  T = eltype(v.data)
  hasgrad(w) && BLAS.gemm!('N', 'T', T(1), v.grad, x.data, T(1), w.grad)
  hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.data, v.grad, T(1), x.grad)
end
