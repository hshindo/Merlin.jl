export LSTM

"""
    LSTM(T::Type, size::Int)

Long short-term memory (LSTM).

```julia
T = Float32
lstm = LSTM(T, 100)
x = Var(rand(T,100,30))
lstm(x)
```
"""
type LSTM
    W
    b
    c0
    h0
end

function LSTM{T}(::Type{T}, size::Int; h0=nothing)
    W = zerograd(uniform(T,-0.001,0.001,size*4,size*2))
    #U = zerograd(orthogonal(T,))
    b = zerograd(zeros(T,size*4))
    b.data[1:size] = ones(T, size) # forget gate initializes to 1
    c0 = Var(zeros(T,size))
    h0 == nothing && (h0 = Var(zeros(T,size)))
    LSTM(W, b, c0, h0)
end

function (lstm::LSTM)(x::Var)
    ndims(x.data) == 2 || throw("x must be matrix.")
    s = size(x.data, 1)
    s*4 == length(lstm.b.data) || throw("Length mismatch.")

    T = eltype(lstm.W.data)
    hs = Var[]
    c = lstm.c0
    h = lstm.h0
    for i = 1:size(x.data,2)
        xi = x[:,i]
        a = lstm.W * cat(1,xi,h) + lstm.b
        f = sigmoid(a[1:s])
        i = sigmoid(a[s+1:2s])
        o = sigmoid(a[2s+1:3s])
        c = f .* c + i .* tanh(a[3s+1:4s])
        h = o .* tanh(c)
        push!(hs, h)
    end
    cat(2, hs...)
end
