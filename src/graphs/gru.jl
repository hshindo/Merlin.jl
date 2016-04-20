export GRU

"""
Gated Recurrent Unit (GRU)
Ref: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

### Functions
- `GRU{T}(::Type{T}, xsize::Int)`
-- xsize: size of input vector (= size of hidden vector)

### 👉 Example
```julia
xs = [rand(Float32,10,5) for i=1:10]
f = GRU(Float32,10)
h = ones(x)
for x in xs
  h = f(x,h)
end
```
"""
function GRU{T}(::Type{T}, xsize::Int)
  # parameters
  Ws = [Variable(rand(T,xsize,xsize)) for i=1:3]
  Us = [Variable(rand(T,xsize,xsize)) for i=1:3]
  # input
  x = Variable()
  h = Variable()

  r = Sigmoid()(Ws[1]*x + Us[1]*h)
  z = Sigmoid()(Ws[2]*x + Us[2]*h)
  h_ = Tanh()(Ws[3]*x + Us[3]*(r.*h))
  h_next = (1 - z) .* h + z .* h_
  Graph(h_next)
end
