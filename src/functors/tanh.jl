type Tanh <: Functor
end

function forward!(f::Tanh, v::Variable)
  v.value = tanh(v[1].value)
end

function backward!(f::Tanh, v::Variable)
  gx = ∇tanh(v.value, v.grad)
  addgrad!(v[1], gx)
end

function ∇tanh{T,N}(f::Tanh, y::Array{T,N}, gy::Array{T,N})
  gx = similar(y)
  for i = 1:length(gx)
    gx[i] = gy[i] * (T(1) - y[i] * y[i])
  end
  gx
end
