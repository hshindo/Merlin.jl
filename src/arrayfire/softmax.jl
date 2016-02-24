type Softmax <: Functor
end

function forward!(f::Softmax, v::Variable)
  logq, y = crossentropy(v[1].value, v[2].value)
  v.state = logq
  v.value = y
end

function softmax{T,N}(p::Array{T,N}, q::Array{T,N})
  logq = logsoftmax(q)
  y = similar(p)
  for i = 1:length(y)
    y[i] = -p[i] * logq[i]
  end
  logq, y
end
