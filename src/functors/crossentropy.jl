type CrossEntropy <: Functor
  p::AFArray
end

CrossEntropy(p::Matrix) = CrossEntropy(AFArray(p))

function forward!(f::CrossEntropy, v::Variable)
  x = v[1].value
  logx = logsoftmax(x)
  v.value = f.p .* logx
  v.work = logx
end

function crossentropy(p::AFArray, x::AFArray)
  logx = logsoftmax(x)
  p .* logx
end

function crossentropy{T,N}(p::Array{T,N}, x::Array{T,N})
  logq = logsoftmax(x)
  y = similar(p)
  for i = 1:length(y)
    y[i] = -p[i] * logq[i]
  end
  logq, y
end

function backward!(f::CrossEntropy, v::Variable)
  x = v[1].value
  logx = v.work
  gx = v.grad .* (exp(logx) + f.p)
  #gx = zeros(x)
  addgrad!(v[1], gx)
end

function ∇crossentropy{T,N}(p::Array{T,N}, q::Array{T,N}, logq::Array{T,N}, gy::Array{T,N})
  gq = alloc_cpu(T, size(q))
  for i = 1:length(gq)
    gq[i] = gy[i] * (exp(logq[i]) - p[i])
  end
  gq
end
