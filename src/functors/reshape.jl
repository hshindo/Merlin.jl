export Reshape

"""
## 🔨 Reshape
Reshapes an array with the given dimensions.

### Functions
- `Reshape(dims::Int...)`

### 👉 Example
```julia
#x = Variable(rand(Float32,10,5,3))
#f = Reshape(5,3,10)
#y = f(x)
```
"""
type Reshape <: Functor
end

