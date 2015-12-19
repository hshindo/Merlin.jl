type CrossEntropy{T} <: Functor
  param::Matrix{T}
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

function apply{T}(f::CrossEntropy{T}, input::Array{T})
  length(f.param) == length(input) || error("CrossEntropy: length unmatch")
  param = f.param
  output = similar(input)
  logp = logsoftmax(input)
  for i = 1:length(input)
    output[i] = -param[i] * logp[i]
  end
  output, gy -> diff(f, input, logp, gy)
end

function diff{T}(f::CrossEntropy{T}, input::Matrix{T}, logp::Matrix{T}, gradout::Matrix{T})
  gradin = similar(input)
  param = f.param
  for i = 1:length(gradin)
    gradin[i] = gradout[i] * (exp(logp[i]) - param[i])
  end
  gradin
end
