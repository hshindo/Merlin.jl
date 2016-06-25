export dropout

"""
    dropout(x, ratio)

Dropout function.
"""
dropout(x::Var, ratio::Float64) = Dropout(ratio)(x)

type Dropout
  ratio::Float64
end

@compat function (f::Dropout)(x::Var)
  @checkargs f (x,)
  throw("Not implemented yet.")
end

function dropout{T,N}(x::Array{T,N}, ratio::Float64)
  scale = T(1.0 / (1.0-ratio))
  randx = rand(Float32, size(x))
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = ifelse(randx[i] > ratio, scale*x[i], T(0))
  end
end

function dropout{T,N}(x::CuArray{T,N}, ratio::Float64)

end

function ∇dropout!(randx::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(randx[i] > ratio, scale*gy[i], T(0))
  end
end
