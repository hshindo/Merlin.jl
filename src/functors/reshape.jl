import Base.reshape

"""
    reshape(x::Var, dims::Int...)
    reshape(x::Var, dims::Tuple)

Reshape an array according to the given dimensions.

### 👉 Example
```julia
x = Var(rand(Float32,10,5,3))
y = reshape(x, 5, 3, 5, 2)
```
"""
reshape{N}(x::Var, dims::NTuple{N,Int}) = Reshape(dims)(x)
reshape(x::Var, dims::Int...) = reshape(x, dims)

type Reshape
  dims
end

@compat function (f::Reshape)(x::Var)
  @checkargs f (x,)
  y = copy(reshape(x.value,f.dims))
  df(gy) = hasgrad(x) && BLAS.axpy!(eltype(gy)(1), gy, x.grad)
  Var(y, df, [x])
end
