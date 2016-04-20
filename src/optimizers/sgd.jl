export SGD

"""
## SGD
Computes Stochastic Gradient Descent.
After updated, gradient is set to be zero.

### Functions
- `SGD(rate::Float64)`

### 👉 Example
```julia
opt = SGD(0.001)
f = Linear(Float32,100,50)
# compute gradient...

update!(opt, f) # update parameters of `f`
update!(opt, param, grad) # param -= rate * grad
```
"""
type SGD <: Optimizer
  rate::Float64 # learning rate
end

function update!{T}(opt::SGD, param::Array{T}, grad::Array{T})
  axpy!(-T(opt.rate), grad, param)
  fill!(grad, T(0.0))
end
