type ReLU <: Functor
  alpha::Float64
end

ReLU() = ReLU(0.0)

function apply{T,N}(fun::ReLU, inputs::Tuple{Array{T,N}})
  input = inputs[1]
  output = similar(input)
  for i = 1:length(input)
    x = input[i]
    output[i] = x > T(0.0) ? x : T(fun.alpha) * x
  end
  output, nothing
end

function diff{T,N}(fun::ReLU, inputs::Tuple{Array{T,N}}, work, gradout::Array{T,N})
  input = inputs[1]
  gradin = similar(input)
  for i = 1:length(input)
    d = input[i] > T(0.0) ? T(1.0) : T(fun.alpha)
    gradin[i] = gradout[i] * d
  end
  tuple(gradin)
end
