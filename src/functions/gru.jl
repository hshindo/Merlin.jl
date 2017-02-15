export GRU

"""
    GRU(T::Type, xsize::Int)

Gated Recurrent Unit (GRU).
See: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

* xsize: size of input vector (= size of hidden vector)

```julia
T = Float32
gru = GRU(T, 100)
x = Var(rand(T,100))
h = Var(rand(T,100))
y = gru(x, h)
```
"""
type GRU
    Ws
    Us
    bs
end

function GRU{T}(::Type{T}, size::Int)
    Ws = (zerograd(uniform(T,-0.01,0.01,size*2,size)), zerograd(uniform(T,-0.01,0.01,size,size)))
    Us = (zerograd(uniform(T,-0.01,0.01,size*2,size)), zerograd(uniform(T,-0.01,0.01,size,size)))
    bs = (zerograd(zeros(T,size*2)), zerograd(zeros(T,size)))
    GRU(Ws, Us, bs)
end

function (f::GRU)(x::Var, h::Var)
    (ndims(x.data) == 1 && ndims(h.data) == 1) || throw("x and h must be vectors.")
    n = length(x.data)
    o = sigmoid(f.Ws[1]*x + f.Us[1]*h + f.bs[1])
    z = o[1:n]
    r = o[n+1:2n]
    z.*h + (1-z) .* tanh(f.Ws[2]*x + f.Us[2]*(r.*h) + f.bs[2])
end
