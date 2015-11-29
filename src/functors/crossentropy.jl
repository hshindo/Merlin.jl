type CrossEntropy{T} <: Functor
  param::Array{T}
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

function apply{T}(fun::CrossEntropy{T}, input::Matrix{T})
  @assert(length(fun.param) == length(input))
  output = similar(input)
  logp = logsoftmax(input)
  for i = 1:length(input)
    output[i] = -fun.param[i] * logp[i]
  end
  output, logp
end

function diff{T}(fun::CrossEntropy{T}, input::Matrix{T}, logp::Matrix{T}, gradout::Matrix{T})
  gradin = similar(input)
  for i = 1:length(input)
    gradin[i] = gradout[i] * (exp(logp[i]) - fun.param[i])
  end
  gradin
end
