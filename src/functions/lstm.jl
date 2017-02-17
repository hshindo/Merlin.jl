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
end

function LSTM{T}(::Type{T}, size::Int)
    W = zerograd(uniform(T,-0.001,0.001,size*4,size*2))
    #U = zerograd(orthogonal(T,))
    b = zerograd(zeros(T,size*4))
    b.data[1:size] = ones(T, size) # forget gate initializes to 1
    LSTM(W, b)
end

function (lstm::LSTM)(x::Var)
    ndims(x.data) == 2 || throw("")
    s = size(x.data, 1)
    s*4 == length(lstm.b.data) || throw("")

    T = eltype(lstm.W.data)
    cs = [Var(rand(T,s))]
    hs = [Var(rand(T,s))]
    for i = 1:size(x.data,2)
        xx = x[(i-1)*s+1:i*s]
        a = lstm.W * cat(1,xx,hs[end]) + lstm.b
        f = sigmoid(a[1:s])
        i = sigmoid(a[s+1:2s])
        o = sigmoid(a[2s+1:3s])
        c = f .* cs[end] + i .* tanh(a[3s+1:4s])
        h = o .* tanh(c)
        push!(cs, c)
        push!(hs, h)
    end
    cat(2, hs...)
end
