function logsoftmax(x::AFArray)
  #m = maximum(x, 1)
  #mm = x - m
  #e = exp(m)
  #z = sum(e, 1)
  #l = log(z)
  #y = mm - l
  #for a in (m,mm,e,z,l)
  #  release(a)
  #end
  #y
  m = x - maximum(x, 1)
  z = sum(exp(m), 1)
  m - log(z)
end

function logsoftmax2(x::AFArray)
  m = maximum(x, 1)
  mm = x - m
  e = exp(m)
  z = sum(e, 1)
  l = log(z)
  y = mm - l
  for a in (m,mm,e,z,l)
    finalize(a)
  end
  y
end

function onehot{T}(x::Vector{Int}, ::Type{T}, size::Int)
  y = zeros(T, size, length(x))
  for j = 1:length(x)
    y[x[j],j] = T(-1)
  end
  AFArray(y)
end
