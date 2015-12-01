type CrossEntropy <: Functor
  param
  logp
end

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

function apply{T}(fun::CrossEntropy, input::Matrix{T})
  length(fun.param) == length(input) || error("CrossEntropy length unmatch")
  param::Matrix{T} = fun.param
  output = similar(input)
  logp = logsoftmax(input)
  for i = 1:length(input)
    output[i] = -param[i] * fun.logp[i]
  end
  fun.logp = logp
  output
end

function diff{T}(fun::CrossEntropy, gradout::Matrix{T}, input::Matrix{T})
  gradin = similar(inputs[1])
  param::Matrix{T} = fun.param
  for i = 1:length(input)
    gradin[i] = gradout[i] * (exp(logp[i]) - param[i])
  end
  tuple(gradin)
end
