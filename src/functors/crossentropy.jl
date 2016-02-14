type CrossEntropy <: Functor
  p::AFArray
end

# Instantiation with one-hot vector
function CrossEntropy(indices::Vector{Int})
  p = zeros(T, len, length(indices))
  for j = 1:length(indices)
    p[indices[j], j] = value
  end
  CrossEntropy(AFArray(p))
end

function forward!(f::CrossEntropy, v::Variable)
  logx = logsoftmax(v[1].value)
  v.work = logx
  v.value = -f.p * logx
end

function backward!(f::CrossEntropy, v::Variable)
  logx = v.work
  gx = v.grad * (exp(logx) - f.p)
  addgrad!(v[1], gx)
end

function ∇crossentropy{T,N}(p::Array{T,N}, q::Array{T,N}, logq::Array{T,N}, gy::Array{T,N})
  gq = alloc_cpu(T, size(q))
  for i = 1:length(gq)
    gq[i] = gy[i] * (exp(logq[i]) - p[i])
  end
  gq
end
