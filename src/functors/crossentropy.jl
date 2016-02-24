type CrossEntropy <: Functor
end

function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(v[1].value)
  y = crossentropy(v[1].value, logq)
  v.work = logq
  v.value = y
end

function crossentropy{T,N}(p::Array{T,N}, logq::Array{T,N})
  y = similar(p)
  for i = 1:length(y)
    y[i] = -p[i] * logq[i]
  end
  y
end

function backward!(f::CrossEntropy, v::Variable)
  gq = ∇crossentropy(v[1].value, v[2].value, v.work, v.grad)
  addgrad!(v[2], gq)
end

function ∇crossentropy{T,N}(p::Array{T,N}, q::Array{T,N}, logq::Array{T,N}, gy::Array{T,N})
  gq = Array(T, size(q))
  for i = 1:length(gq)
    gq[i] = gy[i] * (exp(logq[i]) - p[i])
  end
  gq
end


