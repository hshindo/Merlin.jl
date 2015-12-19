type LogSoftmax
end

function call{T}(fun::LogSoftmax, input::Matrix{T})
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
apply{T}(fun::LogSoftmax, input::Array{T}) = apply(fun, mat(input))

function diff{T}(fun::LogSoftmax, input::Matrix{T}, gradout::Matrix{T})
  # d(y_j) / d(x_i) = delta(i = j) - exp(y_i)
  gradin = similar(input)
  output = Array(T, outsize(input))
  fill!(gradin, T(0.0))
  for d = 1:size(output, 2)
    for i = 1:size(output, 1)
      expy = exp(output[i, d])
      for j = 1:size(output, 1)
        delta = i == j ? T(1.0) : T(0.0)
        gradin[i, d] += gradout[j, d] * (delta - expy)
      end
    end
  end
end
