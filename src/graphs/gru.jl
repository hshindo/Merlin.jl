export GRU

"""
    GRU(::Type, xsize::Int)

Gated Recurrent Unit (GRU).
Ref: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

## Arguments
* xsize: size of input vector (= size of hidden vector)

## Usage
```julia
gru = GRU(Float32,100)
x = Var(rand(Float32,100))
h = Var(rand(Float32,100))
gru(:x=>x, :h=>h)
```
"""
function GRU{T}(::Type{T}, xsize::Int)
  @graph begin
    Ws = [Param(rand(T,xsize,xsize)) for i=1:3]
    Us = [Param(rand(T,xsize,xsize)) for i=1:3]
    x = Data(:x)
    h = Data(:h)
    r = sigmoid(Ws[1]*x + Us[1]*h)
    z = sigmoid(Ws[2]*x + Us[2]*h)
    h_ = tanh(Ws[3]*x + Us[3]*(r.*h))
    h_next = (1 - z) .* h + z .* h_
    h_next
  end
end
