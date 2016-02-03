"""
### Sigmoid function
#### params

#### input
n-d array

#### output
n-d array
"""
type Sigmoid <: Functor
end

function forward{T,N}(f::Sigmoid, x::Array{T,N})
  y = similar(x)
  for i = 1:length(x)
    y[i] = T(1) / (T(1) + exp(-x[i]))
  end
  y, (gy, gx) -> gx == nothing || backward!(f, y, gy, gx)
end

function backward!{T,N}(f::Sigmoid, y::Array{T,N}, gy::Array{T,N}, gx::Array{T,N})
  for i = 1:length(gx)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end
