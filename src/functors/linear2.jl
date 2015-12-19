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

function apply{T}(f::Linear, input::Matrix{T})
  output = Array(T, size(f.weight, 1), size(input, 2))
  gemm!('N', 'N', T(1), f.weight, input, T(0), output)
  broadcast!(+, output, f.bias, output)
  output, gy -> diff(f, input, gy)
end

function diff{T}(fun::Linear{T}, input::Matrix{T}, gradout::Matrix{T})
  gradin = similar(input)
  gemm!('T', 'N', T(1), fun.weight, gradout, T(0), gradin) # d gradout / d input = weight^T * gradout
  gemm!('N', 'T', T(1), gradout, input, T(1), fun.gradweight) # d gradout / d weight = gradout * input^T
  sum!(fun.gradbias, gradout) # d gradout / d bias = 1
  gradin
end

function optimize!(opt::Optimizer, l::Linear)
  update!(opt, l.weight, l.gradweight)
  update!(opt, l.bias, l.gradbias)
end
