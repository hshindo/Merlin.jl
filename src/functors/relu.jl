type ReLU <: Functor
  alpha::Float64
end

ReLU() = ReLU(0.0)

function forward{T,N}(f::ReLU, x::Array{T,N})
  y = similar(x)
  for i = 1:length(x)
    xx = x[i]
    y[i] = xx > T(0) ? xx : T(f.alpha) * xx
  end
  y, (gy, gx) -> gx == nothing || backward!(f, x, gy, gx)
end

function backward!{T,N}(f::ReLU, x::Array{T,N}, gy::Array{T,N}, gx::Array{T,N})
  for i = 1:length(x)
    d = x[i] > T(0) ? gy[i] : T(f.alpha) * gy[i]
    gx[i] += d
  end
end
