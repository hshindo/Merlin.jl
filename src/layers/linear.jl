export Linear

Var(:Linear)

function Linear(T::Type, indim::Int, outdim::Int)
  r = T(sqrt(6 / (indim+outdim)))
  w = rand(-r, r, outdim, indim)
  b = fill(T(0), outdim, 1)
  Linear(nothing, nothing, [Param(w),Param(b),Data()])
end

@compat (l::Linear)(x::Var) = linear(l.w, l.b, x)
@compat (l::Linear)(x::ExprVar) = ExprVar(linear, l.w, l.b, x)

function linear(w::Var, b::Var, x::Var)
  y = w.data * x.data
  broadcast!(.+, y, y, b.data)
  Linear(y, nothing, [w,b,x])
end

function backward!(l::Linear)
  T = eltype(l.data)
  hasgrad(l.w) && BLAS.gemm!('N', 'T', T(1), l.grad, l.x.data, T(1), l.w.grad)
  hasgrad(l.x) && BLAS.gemm!('T', 'N', T(1), l.w.data, l.grad, T(1), l.x.grad)
end
