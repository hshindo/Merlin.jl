function logsoftmax{T}(x::Matrix{T})
  y = similar(x)
  max = maximum(x, 1)
  for j = 1:size(x,2)
    sum = T(0.0)
    @simd for i = 1:size(x,1)
      @inbounds sum += exp(x[i,j] - max[j])
    end
    logz = log(sum)
    @simd for i = 1:size(x,1)
      @inbounds y[i, j] = x[i, j] - max[j] - logz
    end
  end
  y
end
