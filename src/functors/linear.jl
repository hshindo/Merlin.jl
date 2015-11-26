using Base.LinAlg.BLAS

type Linear{T} <: Functor
  weight::Matrix{T}
  bias::Vector{T}
  gradweight::Matrix{T}
  gradbias::Vector{T}
end

Linear(weight, bias) = Linear(weight, bias, zeros(weight), zeros(bias))

function Linear{T}(::Type{T}, inlength::Int, outlength::Int)
  #weight = rand(T, outsize, insize)
  #b = sqrt(6.0 / (insize + outsize))
  #for i = 1:length(weight)
  #  weight[i] = weight[i] * 2b - b
  #end
  weight = convert(Array{T}, randn(outlength, inlength) * sqrt(1 / inlength))
  bias = fill(T(0.0), outlength)
  Linear(weight, bias)
end

mat(a::Array) = reshape(a, size(a, 1), prod(size(a)[2:end]))

function apply{T}(l::Linear{T}, var::Variable)
  data = Array(T, size(l.weight, 1), size(var.data)[2:end]...)
  gemm!('N', 'N', T(1.0), l.weight, var.data, T(0.0), data)
  broadcast!(+, data, l.bias, data)
  Variable(data)
end

function apply{T}(l::Linear{T}, input::Matrix)
  output = Array(T, size(l.weight, 1), size(input)[2:end]...)
  gemm!('N', 'N', T(1.0), l.weight, input, T(0.0), output)
  broadcast!(+, output, l.bias, output)
  output
end

function diff{T}(l::Linear{T}, input::Matrix, gradout::Matrix)
  gradin = similar(input)
  gemm!('T', 'N', T(1.0), l.weight, gradout, T(0.0), gradin) # d gradout / d input = weight^T * gradout
  gemm!('N', 'T', T(1.0), gradout, input, T(1.0), l.gradweight) # d gradout / d weight = gradout * input^T
  sum!(l.gradbias, gradout) # d gradout / d bias = 1
  gradin
end

function optimize!(opt::Optimizer, l::Linear)
  update!(opt, l.weight, l.gradweight)
  update!(opt, l.bias, l.gradbias)
end
