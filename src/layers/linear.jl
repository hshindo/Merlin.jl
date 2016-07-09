export Linear

type Linear <: Layer
  data
  grad
  w
  b
  x
end

function Linear(T::Type, indim::Int, outdim::Int)
  r = T(sqrt(6 / (indim+outdim)))
  w = rand(-r, r, outdim, indim)
  b = fill(T(0), outdim, 1)
  Linear(nothing, nothing, Data(w,zeros(w)), Data(b,zeros(b)), nothing)
end

@compat function (l::Linear)(x::Layer)
  l.x = x
  x.data == nothing || forward!(l)
  l
end

function forward!(l::Linear)
  l.data = l.w.data * l.x.data
  broadcast!(.+, l.data, l.data, l.b.data)
end

tails(l::Linear) = [l.w, l.b, l.x]

function backward!(l::Linear)
  T = eltype(l.data)
  hasgrad(l.w) && BLAS.gemm!('N', 'T', T(1), l.grad, l.x.data, T(1), l.w.grad)
  hasgrad(l.x) && BLAS.gemm!('T', 'N', T(1), l.w.data, l.grad, T(1), l.x.grad)
end
