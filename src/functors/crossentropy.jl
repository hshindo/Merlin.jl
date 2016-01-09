type CrossEntropy <: Functor
end

function forward!(f::CrossEntropy, v::Variable)
  y, logq = crossentropy(v[1].value, v[2].value)
  v.value = y
  v.state = logq
end

function crossentropy{T,N}(p::Array{T,N}, q::Array{T,N})
  logq = logsoftmax(q)
  y = alloc_cpu(T, size(p))
  for i = 1:length(y)
    y[i] = -p[i] * logq[i]
  end
  y, logq
end

function backward!(f::CrossEntropy, v::Variable)
  gq = ∇crossentropy(v[1].value, v[2].value, v.state, v.grad)
  addgrad!(v[2], gq)
end

function ∇crossentropy{T,N}(p::Array{T,N}, q::Array{T,N}, logq::Array{T,N}, gy::Array{T,N})
  gq = alloc_cpu(T, size(q))
  for i = 1:length(gq)
    gq[i] = gy[i] * (exp(logq[i]) - p[i])
  end
  gq
end

function logsoftmax{T}(x::Matrix{T})
  y = alloc_cpu(T, size(x))
  max = maximum(x, 1)
  for j = 1:size(x,2)
    sum = T(0.0)
    for i = 1:size(x,1)
      sum += exp(x[i, j] - max[j])
    end
    logz = log(sum)
    for i = 1:size(x,1)
      y[i, j] = x[i, j] - max[j] - logz
    end
  end
  y
end
