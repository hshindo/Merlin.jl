export GRU

"""
Gated Recurrent Unit (GRU)
Ref: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

### Functions
- `GRU{T}(::Type{T}, xsize::Int)`
- xsize: size of input vector (= size of hidden vector)
"""
function GRU{T}(::Type{T}, xsize::Int)
  Ws = [Var(rand(T,xsize,xsize),grad=true) for i=1:3]
  Us = [Var(rand(T,xsize,xsize),grad=true) for i=1:3]
  x = Var()
  h = Var()
  r = sigmoid(Ws[1]*x + Us[1]*h)
  z = sigmoid(Ws[2]*x + Us[2]*h)
  h_ = tanh(Ws[3]*x + Us[3]*(r.*h))
  h_next = (1 - z) .* h + z .* h_
  Graph(h_next)
end
