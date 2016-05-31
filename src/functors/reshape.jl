type Reshape
  dims
end

"""
Reshape an array according to the given dimensions.

### 👉 Example
```julia
x = Var(rand(Float32,10,5,3))
y = reshape(x, 5, 3, 5, 2)
```
"""
Base.reshape(x::Var, dims::Int...) = forward(Reshape(dims), [x])

forward!(f::Reshape, y::Var) = y.value = copy(reshape(y[1].value, f.dims))

backward!(f::Reshape, y::Var) = hasgrad(y[1]) && ∇reshape!(y[1].grad, y.grad)

∇reshape!{T}(gx::Array{T}, gy::Array{T}) = BLAS.axpy!(T(1), gy, gx)
