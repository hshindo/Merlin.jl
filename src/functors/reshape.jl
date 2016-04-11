export Reshape

"""
## Reshape
Reshapes an array with the given dimensions.

### Functions
- `Reshape(dims::Int...)`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5,3))
f = Reshape(5,3,10)
y = f(x)
```
"""
type Reshape <: Functor
  dims
end

@compat (f::Reshape)(arg) = forward(f, arg)
function forward!(f::Reshape, v::Variable)
  s = size(v[1].value)
  v.value = reshape(v[1].value, f.dims)
  v.backward! = () -> begin
    T = eltype(v.value)
    axpy!(T(1), v.grad, v[1].grad)
  end
end
