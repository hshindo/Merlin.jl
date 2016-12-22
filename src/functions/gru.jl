export GRU

"""
    GRU(T::Type, xsize::Int)

Gated Recurrent Unit (GRU).
See: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

* xsize: size of input vector (= size of hidden vector)

```julia
T = Float32
gru = GRU(T, 100)
x = constant(rand(T,100))
h = Var(rand(T,100))
y = gru(x, h)
```
"""
function GRU{T}(::Type{T}, xsize::Int)
    ws = [Var(rand(T,xsize,xsize)) for i=1:3]
    us = [Var(rand(T,xsize,xsize)) for i=1:3]
    x = Var()
    h = Var()
    r = sigmoid(ws[1]*x + us[1]*h)
    z = sigmoid(ws[2]*x + us[2]*h)
    h_ = tanh(ws[3]*x + us[3]*(r.*h))
    h_next = (1 - z) .* h + z .* h_
    compile(h_next, x, h)
end
