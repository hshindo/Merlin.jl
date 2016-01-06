type CrossEntropy <: Functor
  x
  logq
  y
end

CrossEntropy() = CrossEntropy(nothing, nothing, nothing)

clone(f::CrossEntropy) = CrossEntropy()

function forward!(f::CrossEntropy)
  p, q = f.x
  length(p) == length(q) || error("length unmatch")
  f.logq == nothing && (f.logq = default(q))
  logq = resize!(f.logq, size(q))
  f.y == nothing && (f.y = default(q))
  y = resize!(f.y, size(q))
  crossentropy(p.value, q.value, logq.value, y.value)
end

function crossentropy!{T}(p::Matrix{T}, q::Matrix{T}, logq::Matrix{T}, y::Matrix{T})
  logsoftmax!(q, logq)
  for i = 1:length(y)
    y[i] = -p[i] * logq[i]
  end
end

backward!(f::CrossEntropy) = ∇crossentropy(f.x[1], f.x[2], f.logq, f.y)

function ∇crossentropy!{T}(varp::Var{T,2}, varq::Var{T,2}, logq::Matrix{T}, vary::Var{T,2})
  p, gp = data(varp)
  q, gq = data(varq)
  y, gy = data(vary)
  for i = 1:length(gx)
    gx[i] += gy[i] * (exp(logq[i]) - p[i])
  end
end

function logsoftmax!{T}(x::Matrix{T}, y::Matrix{T})
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
end
