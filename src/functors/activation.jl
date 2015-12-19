"""
Rectifier Linear Unit
"""
type ReLU <: Functor
  alpha::Float64
end

ReLU() = ReLU(0.0)

function apply{T,N}(f::ReLU, x::Array{T,N})
  y = similar(x)
  for i = 1:length(x)
    xx = x[i]
    y[i] = xx > T(0) ? xx : T(f.alpha) * xx
  end
  y, gy -> diff(f, x, gy)
end

function diff{T,N}(f::ReLU, x::Array{T,N}, gy::Array{T,N})
  gx = similar(x)
  for i = 1:length(x)
    d = x[i] > T(0) ? T(1) : T(f.alpha)
    gx[i] = gy[i] * d
  end
  gx
end

"""
Tanh function
"""
type Tanh <: Functor
end

function apply{T,N}(f::Tanh, x::Array{T,N})
  y = similar(x)
  for i = 1:length(x)
    y[i] = tanh(x[i])
  end
  y, gy -> diff(f, y, gy)
end

function diff{T,N}(f::Tanh, y::Array{T,N}, gy::Array{T,N})
  gx = similar(x)
  for i = 1:length(gx)
    gx[i] = gy[i] * (T(1) - y[i] * y[i])
  end
  gx
end

"""
Sigmoid function
"""
type Sigmoid <: Functor
end

function apply{T,N}(f::Sigmoid, x::Array{T,N})
  y = similar(x)
  for i = 1:length(x)
    y[i] = T(1) / (T(1) + exp(-x[i]))
  end
  y, gy -> diff(f, y, gy)
end

function diff{T,N}(f::Sigmoid, y::Array{T,N}, gy::Array{T,N})
  gx = similar(x)
  for i = 1:length(gx)
    gx[i] = gy[i] * y[i] * (T(1) - y[i])
  end
  gx
end
