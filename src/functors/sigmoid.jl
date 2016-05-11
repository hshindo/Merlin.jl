export Sigmoid

"""
## Sigmoid

- `Sigmoid()`
- `sigmoid()`
"""
type Sigmoid <: Functor
end

@compat function (f::Sigmoid)(xs::Vector{Var})
  x = xs[1]
  y = tanh(x.val * 0.5) * 0.5 + 0.5
  backward! = gy -> hasgrad(x) && ∇sigmoid!(y, x.grad, gy)
  Var(y, nothing, f, xs, backward!)
end
@compat (f::Sigmoid)(x::Var) = f([x])

function ∇sigmoid!{T,N}(y::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(y)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end
