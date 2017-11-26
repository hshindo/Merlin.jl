export BiLSTM

doc"""
    BiLSTM(x)

Bidirectional Long Short-Term Memory network.

# 👉 Example
```julia
x = Var(rand(Float32,10,5))
```
"""
struct BiLSTM
    rnn1::LSTM
    rnn2::LSTM
end

function BiLSTM(::Type{T}, insize::Int, outsize::Int) where T
    BiLSTM(LSTM(T,insize,outsize), LSTM(T,insize,outsize))
end

function (bilstm::BiLSTM)(x::Var)
    lstm = bilstm.rnn1
    h, c = lstm.h0, lstm.c0
    n = size(x.data, 1)
    hs = Var[]
    for i = 1:size(x.data,2)
        h, c = lstm(x[(i-1)*n+1:i*n], h, c)
        push!(hs, h)
    end
    y1 = cat(2, hs...)

    lstm = bilstm.rnn2
    h, c = lstm.h0, lstm.c0
    hs = Var[]
    for i = size(x.data,2):-1:1
        h, c = lstm(x[(i-1)*n+1:i*n], h, c)
        push!(hs, h)
    end
    reverse!(hs)
    y2 = cat(2, hs...)
    cat(1, y1, y2)
end

function (bilstm::BiLSTM)(xs::Vector{Var})
    h1 = bilstm.rnn1(xs)
    h2 = bilstm.rnn2(reverse(xs))
    ys = Array{Var}(length(xs))
    for i = 1:length(xs)
        ys[i] = cat(1, h1[i], h2[i])
    end
    ys
end
