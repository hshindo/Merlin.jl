type CrossEntropy <: Functor
  param::Array
end

CrossEntropy(param::Array) = CrossEntropy(param, eltype(param)[])

function logsoftmax{T}(input::Matrix{T})
  output = similar(input)
  max = maximum(input, 1)
  for j = 1:size(input, 2)
    sum = T(0.0)
    for i = 1:size(input, 1)
      sum += exp(input[i, j] - max[j])
    end
    logz = log(sum)
    for i = 1:size(input, 1)
      output[i, j] = input[i, j] - max[j] - logz
    end
  end
  output
end

function apply(fun::CrossEntropy, var::Variable)
  data = similar(var.data)
  logp = logsoftmax(var.data)
  for i = 1:length(var.data)
    data[i] = -fun.param[i] * logp[i]
  end
  Variable(data, logp)
end

function apply(fun::CrossEntropy, input::Matrix)
  @assert(length(fun.param) == length(input))
  output = similar(input)
  logp = logsoftmax(input)
  for i = 1:length(input)
    output[i] = -fun.param[i] * fun.logp[i]
  end
  output, logp
end

function diff(fun::CrossEntropy, input::Matrix, gradout::Matrix)
  gradin = similar(input)
  for i = 1:length(input)
    gradin[i] = gradout[i] * (exp(fun.logp[i]) - fun.param[i])
  end
  gradin
end
