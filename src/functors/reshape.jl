import Base.reshape

"""
    reshape(x::Var, dims::Int...)

Reshape an array according to the given dimensions.

### 👉 Example
```julia
x = Var(rand(Float32,10,5,3))
y = reshape(x, 5, 3, 5, 2)
```
"""
function reshape(x::Var, dims::Int...)
  y = copy(reshape(x.value, dims))
  f(gy) = hasgrad(x) && ∇reshape!(x.grad, gy)
  Var(y, nothing, f, [x])
end

∇reshape!{T}(gx::Array{T}, gy::Array{T}) = BLAS.axpy!(T(1), gy, gx)
