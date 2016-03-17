type CrossEntropy <: Functor
  p::Matrix
end

function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(v[1].value)
  y = crossentropy(f.p, logq)
  v.value = y
  v.backward! = () -> ∇crossentropy!(f.p, logq, v[1].grad, v.grad)
end

function crossentropy{T}(p::Matrix{T}, logq::Matrix{T})
  y = similar(p)
  for i = 1:length(y)
    y[i] = -p[i] * logq[i]
  end
  y
end

function ∇crossentropy!{T}(p::Matrix{T}, logq::Matrix{T}, gq::Matrix{T}, gy::Matrix{T})
  for i = 1:length(gq)
    gq[i] += gy[i] * (exp(logq[i]) - p[i])
  end
end
