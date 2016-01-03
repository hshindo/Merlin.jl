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

function forward{T}(f::CrossEntropy{T}, x::Array{T})
  length(f.param) == length(x) || error("CrossEntropy: length unmatch")
  param = f.param
  y = similar(x)
  logp = logsoftmax(x)
  for i = 1:length(y)
    y[i] = -param[i] * logp[i]
  end
  y, (gy, gx) -> gx == nothing || backward!(f, logp, gy, gx)
end

function backward!{T}(f::CrossEntropy{T}, logp::Matrix{T}, gy::Matrix{T}, gx::Matrix{T})
  param = f.param
  for i = 1:length(gx)
    gx[i] += gy[i] * (exp(logp[i]) - param[i])
  end
end
