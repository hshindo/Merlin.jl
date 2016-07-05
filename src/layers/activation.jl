export relu

type ReLU <: Layer
  x::Layer
  y
  gy
end

function relu(x)
  y = typeof(x.y) <: Symbol ? Symbol() : relu(x.y)
  ReLU(x, y, nothing)
end

tails(l::ReLU) = [l.x]
backward!(l::ReLU) = hasgrad(x) && ∇relu!(l.x.y, l.x.gy, l.y, l.gy)

function relu{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
  end
end
