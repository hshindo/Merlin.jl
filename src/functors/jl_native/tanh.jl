"""
### Tanh function
#### params

#### input
n-d array

#### output
n-d array
"""
type Tanh <: Functor
end

function forward{T,N}(f::Tanh, x::Array{T,N})
  y = similar(x)
  for i = 1:length(x)
    y[i] = tanh(x[i])
  end
  y, (gy, gx) -> gx == nothing || backward!(f, y, gy, gx)
end

function backward!{T,N}(f::Tanh, y::Array{T,N}, gy::Array{T,N}, gx::Array{T,N})
  for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end
