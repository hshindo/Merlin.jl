type CrossEntropy <: Functor
  p
end

function onehot{T}(x::Vector{Int}, ::Type{T}, size::Int)
  y = zeros(T, size, length(x))
  for j = 1:length(x)
    y[x[j],j] = T(-1)
  end
  AFArray(y)
end

function forward!(f::CrossEntropy, v::Variable)
  x = v[1].value
  T = eltype(x)
  logx = logsoftmax(x)
  conv(p::Vector{Int}) = onehot(f.p, eltype(x), size(x,1))
  conv(p::Matrix{T}) = AFArray(p)
  conv(p::AFArray) = p
  f.p = conv(f.p)
  v.work = logx
  v.value = -f.p * logx
  v.f = CrossEntropy()
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
