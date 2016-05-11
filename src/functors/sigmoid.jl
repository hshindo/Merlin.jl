export Sigmoid

"""
## Sigmoid

- `Sigmoid()`
- `sigmoid()`
"""
type Sigmoid <: Functor
end

function forward(f::Sigmoid, args::Vector{Var})
  x = args[1]
  y = tanh(x.val * 0.5) * 0.5 + 0.5
  backward! = gy -> hasgrad(x) && ∇sigmoid!(y, x.grad, gy)
  Var(y, nothing, f, args, backward!)
end

function ∇sigmoid!{T,N}(y::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(y)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end
