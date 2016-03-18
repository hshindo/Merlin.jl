type Tanh <: Functor
end

function forward!(f::Tanh, v::Variable)
  v.value = tanh(v[1].value)
  v.backward! = () -> ∇tanh!(v[1].grad, v.value, v.grad)
end

function ∇tanh!{T,N}(gx::Array{T,N}, y::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end
