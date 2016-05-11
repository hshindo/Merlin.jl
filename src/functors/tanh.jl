export Tanh

"""
## Tanh

- `Tanh()`
- `tanh()`
"""
type Tanh <: Functor
end

@compat function (f::Tanh)(xs::Vector{Var})
  x = xs[1]
  y = tanh(x.val)
  backward! = gy -> hasgrad(x) && ∇tanh!(y, x.grad, gy)
  Var(y, nothing, f, xs, backward!)
end
@compat (f::Tanh)(x::Var) = f([x])

function ∇tanh!{T,N}(y::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end
