type ReLU <: Functor
  alpha::Float64
end

ReLU() = ReLU(0.0)

function apply(fun::ReLU, var::Variable)
  input = var.data
  output = similar(input)
  for i = 1:length(input)
    x = input[i]
    output[i] = x > 0.0 ? x : fun.alpha * x
  end
  Variable(output)
end

function apply(fun::ReLU, input::Array)
  output = similar(input)
  for i = 1:length(input)
    x = input[i]
    output[i] = x > 0.0 ? x : fun.alpha * x
  end
  (output,)
end

function diff(fun::ReLU, input::Array, gradout::Array)
  gradin = similar(input)
  for i = 1:length(input)
    d = input[i] > 0.0 ? 1.0 : fun.alpha
    gradin[i] = gradout[i] * d
  end
  gradin
end
