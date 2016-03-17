type CrossEntropy <: Functor
  p
end

function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(v[1].value)
  v.value = crossentropy(f.p, logq)
  v.backward! = () -> ∇crossentropy!(f.p, logq, v[1].grad, v.grad)
end

function crossentropy{T}(p::Matrix{T}, logq::Matrix{T})
  y = Array(T, 1, size(p,2))
  for j = 1:size(p,2)
    s = T(0)
    @simd for i = 1:size(p,1)
      @inbounds s += -p[i,j] * logq[i,j]
    end
    y[j] = s
  end
  y
end

function ∇crossentropy!{T}(p::Matrix{T}, logq::Matrix{T}, gq::Matrix{T}, gy::Matrix{T})
  #@simd for i = 1:length(gq)
  #  gq[i] += gy[i] * (exp(logq[i]) - p[i])
  #end
  for j = 1:size(p,2)
    g = gy[j]
    @simd for i = 1:size(p,1)
      @inbounds gq[i,j] += g * (exp(logq[i,j]) - p[i,j])
    end
  end
end
