function logsoftmax(x::AFArray)
  m = x - maximum(x, 1)
  z = sum(exp(m), 1)
  m - log(z)
end
