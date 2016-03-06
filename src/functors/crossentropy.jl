type CrossEntropy <: Functor
  p::Matrix
end

function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(v[1].value)
  y = crossentropy(f.p, logq)
  v.work = logq
  v.value = y
end

function crossentropy{T}(p::Matrix{T}, logq::Matrix{T})
  #y = similar(p)
  y = alloc_cpu(p)
  for i = 1:length(y)
    y[i] = -p[i] * logq[i]
  end
  y
end

function backward!(f::CrossEntropy, v::Variable)
  logq = v.work
  gq = ∇crossentropy(f.p, logq, v.grad)
  addgrad!(v[1], gq)
end

function ∇crossentropy{T}(p::Matrix{T}, logq::Matrix{T}, gy::Matrix{T})
  #gq = similar(p)
  gq = alloc_cpu(p)
  for i = 1:length(gq)
    gq[i] = gy[i] * (exp(logq[i]) - p[i])
  end
  gq
end
