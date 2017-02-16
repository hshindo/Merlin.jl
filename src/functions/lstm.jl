export LSTM

"""
    LSTM(T::Type, size::Int)

Long short-term memory (LSTM).

* size: size of input vector

```julia
T = Float32
lstm = LSTM(T, 100)
c
h
x = Var(rand(T,100))
c, h = lstm(c, h, x)
```
"""
type LSTM
    W
end

function LSTM{T}(::Type{T}, size::Int)
    W = zerograd(uniform(T,-0.001,0.001,size*4,size*2))
    LSTM(W)
end

function (f::LSTM)(c::Var, h::Var, x::Var)
    a = W * cat(1,x,h) .+ b
    f = sigmoid(a[1:n])
    i = sigmoid(a[n+1:2n])
    o = sigmoid(a[2n+1:3n])
    c = f .* c_prev + i .* tanh(a[3n+1:4n])
    h = o .* tanh(c)
    c, h
end
