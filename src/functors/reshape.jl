import Base.reshape

type Reshape
  dims
end

@compat function (f::Reshape)(args::Vector{Var})
  x = args[1]
  y = copy(reshape(x.value, f.dims))
  df(gy) = hasgrad(x) && ∇reshape!(x.grad, gy)
  Var(y, df, [x])
end

"""
    reshape(x::Var, dims::Int...)
    reshape(x::Var, dims::Tuple{Vararg{Int}})

Reshape an array according to the given dimensions.

### 👉 Example
```julia
x = Var(rand(Float32,10,5,3))
y = reshape(x, 5, 3, 5, 2)
```
"""
reshape(x::Var, dims::Tuple{Vararg{Int}}) = forward(Reshape(dims), [x])
reshape(x::Var, dims::Int...) = reshape(x, dims)

∇reshape!{T}(gx::Array{T}, gy::Array{T}) = BLAS.axpy!(T(1), gy, gx)
