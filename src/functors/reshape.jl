export Reshape

"""
## Reshape
Reshapes an array with the given dimensions.

### Functions
- `Reshape(dims::Int...)`

### 👉 Example
```julia
x = Var(rand(Float32,10,5,3))
f = Reshape(5,3,10)
y = f(x)
```
"""
type Reshape <: Functor
  dims
end

Reshape(dims::Int...) = Reshape(dims)

function forward(f::Reshape, args::Vector{Var})
  x = args[1]
  s = size(x.val)
  y = copy(reshape(x.val, f.dims))
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x) && BLAS.axpy!(T(1), gy, x.grad)
  end
  Var(y, nothing, f, args, backward!)
end
