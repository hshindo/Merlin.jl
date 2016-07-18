export Linear, linear

@Var(Linear)

function Linear(T::Type, indim::Int, outdim::Int)
  r = T(sqrt(6 / (indim+outdim)))
  w = rand(-r, r, outdim, indim)
  b = fill(T(0), outdim, 1)
  Linear(nothing, nothing, [Param(w),Data(),Param(b)])
end

@compat (f::Linear)(x::Var) = linear(f[1], x, f[3])
@compat (::Linear)(w::Var, b::Var, x::Var) = linear(w, x, b)

function linear(w::Var, x::Var, b::Var)
  !hasdata(w) || !hasdata(b) || !hasdata(x) && return Linear(nothing, nothing, [w,x,b])
  y = w.data * x.data
  broadcast!(.+, y, y, b.data)
  Linear(y, nothing, [w,x,b])
end

function backward!(v::Linear)
  ∇linear!(v)
  #w, b, x = v[1], v[2], v[3]
  #T = eltype(v.data)
  #hasgrad(w) && BLAS.gemm!('N', 'T', T(1), v.grad, x.data, T(1), w.grad)
  #hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.data, v.grad, T(1), x.grad)
end

function ∇linear!(y::Var)
  T = eltype(y.data)
  hasgrad(y[1]) && BLAS.gemm!('N', 'T', T(1), y.grad, y[2].data, T(1), y[1].grad)
  hasgrad(y[2]) && BLAS.gemm!('T', 'N', T(1), y[1].data, y.grad, T(1), y[2].grad)
  length(y.tails) == 2 && return
  for offset = 1:length(y[3].data):length(y.grad)
    BLAS.axpy!(length(y[3].data), T(1), pointer(y.grad,offset), 1, pointer(y[3].grad), 1)
  end
end
