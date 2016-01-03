using Base.LinAlg.BLAS

type Linear{T} <: Functor
  weight::Matrix{T}
  bias::Vector{T}
  gradweight::Matrix{T}
  gradbias::Vector{T}
end

Linear(weight, bias) = Linear(weight, bias, zeros(weight), zeros(bias))

function Linear{T}(::Type{T}, inlen::Int, outlen::Int)
  weight = convert(Array{T}, randn(outlen, inlen) * sqrt(1 / inlen))
  bias = fill(T(0.01), outlen)
  Linear(weight, bias)
end

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))

isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1

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
