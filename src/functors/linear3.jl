using Base.LinAlg.BLAS

type Linear <: Functor
  weight::Var
  bias::Var
end

function Linear{T}(::Type{T}, inlen::Int, outlen::Int)
  w = convert(Array{T}, randn(outlen, inlen) * sqrt(1 / inlen))
  b = fill(T(0.01), outlen)
  Linear(Var(w), Var(b))
end

function forward{T}(f::Linear, vx::Var{Matrix{T}})
  w, b, x = f.weight.value, f.bias.value, vx.value
  y = Array(T, size(w,1), size(x,2))
  gemm!('N', 'N', T(1), w, x, T(0), y)
  broadcast!(+, y, b, y)
  vy = Var()
  y, v -> (f, x, v)
end

"""
d gradout / d input = weight^T * gradout
d gradout / d weight = gradout * input^T
d gradout / d bias = 1
"""
function backward!{T}(f::Linear, x::Var{Matrix{T}}, y::Var{Matrix{T}})
  w, b = f.weight, f.bias
  x.fixed || gemm!('T', 'N', T(1), w.value, y.grad, T(1), x.grad)
  w.fixed || gemm!('N', 'T', T(1), y.grad, x.value, T(1), w.grad)
  b.fixed || sum!(b.grad, y.grad)
end

function forward{T}(f::Linear, x::Matrix{T})
  y = Array(T, size(f.weight, 1), size(x, 2))
  gemm!('N', 'N', T(1), f.weight, x, T(0), y)
  broadcast!(+, y, f.bias, y)
  y, (gy, gx) -> gx == nothing || backward!(f, x, gy, gx)
end

function backward!{T}(f::Linear{T}, x::Matrix{T}, gy::Matrix{T}, gx::Matrix{T})
  gemm!('T', 'N', T(1), f.weight, gy, T(1), gx) # d gradout / d input = weight^T * gradout
  gemm!('N', 'T', T(1), gy, x, T(1), f.gradweight) # d gradout / d weight = gradout * input^T
  sum!(f.gradbias, gy) # d gradout / d bias = 1
end

function optimize!(opt::Optimizer, l::Linear)
  update!(opt, l.weight, l.gradweight)
  update!(opt, l.bias, l.gradbias)
end
