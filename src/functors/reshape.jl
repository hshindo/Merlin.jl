import Base.reshape

type Reshape <: Functor
  dims
end

forward{T<:Number}(f::Reshape, x::Array{T}) = f, copy(reshape(x, f.dims))

function backward!{T}(f::Reshape, x::Array{T}, gx, y, gy)
  isempty(gx) && return
  BLAS.axpy!(T(1), gy, gx)
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
reshape{N}(x::Var, dims::NTuple{N,Int}) = forward(Reshape(dims), x)
reshape(x::Var, dims::Int...) = reshape(x, dims)
