export GRU

"""
    GRU(::Type, xsize::Int)

Gated Recurrent Unit (GRU).
See: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

## Arguments
* xsize: size of input vector (= size of hidden vector)

## 👉 Example
```julia
gru = GRU(Float32,100)
x = Var(rand(Float32,100))
h = Var(rand(Float32,100))
y = gru(x, h)
```
"""
function GRU(T::Type, xsize::Int)
    ws = [zerograd!(Var(rand(T,xsize,xsize))) for i=1:3]
    us = [zerograd!(Var(rand(T,xsize,xsize))) for i=1:3]
    @graph begin
        h = :h
        x = :x
        r = sigmoid(ws[1]*x + us[1]*h)
        z = sigmoid(ws[2]*x + us[2]*h)
        h_ = tanh(ws[3]*x + us[3]*(r.*h))
        h_next = (1 - z) .* h + z .* h_
        h_next
    end
end
