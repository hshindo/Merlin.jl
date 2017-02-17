export LSTM

"""
    LSTM(T::Type, size::Int)

Long short-term memory (LSTM).

```julia
T = Float32
lstm = LSTM(T, 100)
c = Var()
h = Var()
x = Var(rand(T,100))
c_next, h_next = lstm(c, h, x)
```
"""
type LSTM
    W
    b
end

function LSTM{T}(::Type{T}, size::Int)
    W = zerograd(uniform(T,-0.001,0.001,size*4,size*2))
    b = zerograd(zeros(T,size*4))
    b.data[1:size] = ones(T, size) # forget gate initializes as one
    LSTM(W, b)
end

function (f::LSTM)(x::Var)
    ndims(x.data) == 2 || throw("")
    n = length(f.b.data) / 4
    c = Var(zeros(eltype(f.W.data),n))
    h = Var(zeros(eltype(f.W.data),n))
    for i = 1:length(x.data)
        a = f.W * cat(1,x,h) + f.b
        f = sigmoid(a[1:n])
        i = sigmoid(a[n+1:2n])
        o = sigmoid(a[2n+1:3n])
        c = f .* c + i .* tanh(a[3n+1:4n])
        h = o .* tanh(c)
    end
end
