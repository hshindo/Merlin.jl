type Sum
  dim::Int
end

"""
Compute the sum along the given dimension.

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = sum(x, 1)
```
"""
Base.sum(dim::Int, x::Var) = forward(Sum(dim), [x])

function forward!(f::Sum, y::Var)
  y.value = sum(y[1].value, f.dim)
  y.f = f
end

backward!(f::Sum, y::Var) = hasgrad(y[1]) && ∇sum!(y[1].grad, y.grad)

function ∇sum!{T,N}(gx::Array{T,N}, gy::Array{T,N})
  broadcast!(+, gx, gx, gy)
end
