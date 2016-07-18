export relu

@Var(ReLU)

"""
    relu(x)
"""
function relu(x::Var)
  y = hasdata(x) ? relu(x.data) : nothing
  ReLU(y, nothing, [x])
end
@compat (::ReLU)(x::Var) = relu(x)

function backward!(v::ReLU)
  hasgrad(v[1]) || return
  ∇relu!(v[1].data, v[1].grad, v.data, v.grad)
end

function relu{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end

relu(x::CuArray) = activation!(CUDNN_ACTIVATION_RELU, x, similar(x))

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
  end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  ∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end
