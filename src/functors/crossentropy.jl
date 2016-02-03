type CrossEntropy <: Functor
end

function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(q)
  v.state = logq
  v.value = -p * logq
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
