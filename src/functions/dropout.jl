export dropout

"""
    dropout(x::Var, ratio::Float64, istrain::Bool)
"""
function dropout(x::Var, ratio::Float64, istrain::Bool)
    istrain || return x
    y = dropout(x.data, ratio)
    df(gy) = gy * self.mask
    Var(y, [x], dropout, df)
end

function dropout{T}(x::Array{T}, ratio::Float64)
    scale = T(1.0 / (1.0-ratio))
    rx = rand(T, length(x))
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = ifelse(rx[i] <= T(ratio), T(0), scale*x[i])
    end
    y
end

function ∇dropout!(ratio::Float64, rx::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(x)
      gx[i] += ifelse(randx[i] <= T(ratio), T(0), scale*gy[i])
  end
  gx
end
