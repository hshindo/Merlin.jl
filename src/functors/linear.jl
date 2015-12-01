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

function apply{T}(fun::Linear{T}, inputs::Tuple{Matrix{T}})
  output = Array(T, size(fun.weight, 1), size(input)[2:end]...)
  gemm!('N', 'N', T(1.0), fun.weight, input, T(0.0), output)
  broadcast!(+, output, fun.bias, output)
  output, nothing
end

function diff{T}(fun::Linear{T}, inputs::Tuple{Matrix{T}}, work, gradout::Matrix{T})
  gradin = similar(inputs[1])
  gemm!('T', 'N', T(1.0), fun.weight, gradout, T(0.0), gradin) # d gradout / d input = weight^T * gradout
  gemm!('N', 'T', T(1.0), gradout, input, T(1.0), fun.gradweight) # d gradout / d weight = gradout * input^T
  sum!(fun.gradbias, gradout) # d gradout / d bias = 1
  tuple(gradin)
end

function optimize!(opt::Optimizer, l::Linear)
  update!(opt, l.weight, l.gradweight)
  update!(opt, l.bias, l.gradbias)
end
