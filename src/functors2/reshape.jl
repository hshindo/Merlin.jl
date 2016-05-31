"""
Reshape an array according to the given dimensions.

## Parameters
- x::Var
- dims::Int...

### 👉 Example
```julia
x = Var(rand(Float32,10,5,3))
y = reshape(x, 5, 3, 5, 2)
```
"""
Base.reshape(x::Var, dims::Int...) = forward0(Reshape(dims), [x])

type Reshape <: Functor
  dims
end

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
