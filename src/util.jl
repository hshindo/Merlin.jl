function logsoftmax(x::AFArray)
  m = x - maximum(x, 1)
  z = sum(exp(m), 1)
  m - log(z)
end

function onehot{T}(x::Vector{Int}, ::Type{T}, size::Int)
  y = zeros(T, size, length(x))
  for j = 1:length(x)
    y[x[j],j] = T(-1)
  end
  AFArray(y)
end
