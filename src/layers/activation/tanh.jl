import Base.tanh

@Var(Tanh)

"""
    tanh(x)
"""
function tanh(x::Var)
  y = hasdata(x) ? tanh(x.data) : nothing
  Tanh(y, nothing, [x])
end
@compat (::Tanh)(x::Var) = tanh(x)

function backward!(v::Tanh)
  hasgrad(v[1]) || return
  ∇tanh!(v[1].data, v[1].grad, v.data, v.grad)
end

tanh(x::CuArray) = activation!(CUDNN_ACTIVATION_TANH, x, similar(x))

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  ∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx, beta=1.0)
end
