type CrossEntropy <: Functor
  p
  q
  logq
  y
end

CrossEntropy() = CrossEntropy(nothing, nothing, nothing, nothing)

function forward!(f::CrossEntropy, p, q)
  length(p) == length(q) || error("length unmatch")
  f.p = p
  f.q = q
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

backward!(f::CrossEntropy) = ∇crossentropy()

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

function forward{T}(f::CrossEntropy{T}, x::Array{T})
  length(f.param) == length(x) || error("CrossEntropy: length unmatch")
  param = f.param
  y = similar(x)
  logp = logsoftmax(x)
  for i = 1:length(y)
    y[i] = -param[i] * logp[i]
  end
  y, (gy, gx) -> gx == nothing || backward!(f, logp, gy, gx)
end

function backward!{T}(f::CrossEntropy{T}, logp::Matrix{T}, gy::Matrix{T}, gx::Matrix{T})
  param = f.param
  for i = 1:length(gx)
    gx[i] += gy[i] * (exp(logp[i]) - param[i])
  end
end
