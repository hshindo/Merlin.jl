type CrossEntropy <: Functor
end

function forward!(f::CrossEntropy, v::Variable)
  p, x = v[1].value, v[2].value
  logx = logsoftmax(x)
  v.value = p .* logx
  v.work = logx
end

function backward!(f::CrossEntropy, v::Variable)
  p, x = v[1].value, v[2]
  logx = v.work
  gx = v.grad .* (exp(logx) + p)
  addgrad!(x, gx)
end

function ∇crossentropy{T,N}(p::Array{T,N}, q::Array{T,N}, logq::Array{T,N}, gy::Array{T,N})
  gq = alloc_cpu(T, size(q))
  for i = 1:length(gq)
    gq[i] = gy[i] * (exp(logq[i]) - p[i])
  end
  gq
end
