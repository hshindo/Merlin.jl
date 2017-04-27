export LSTM, BiLSTM

"""
    LSTM(T::Type, xsize::Int, hsize::Int)

Long short-term memory (LSTM).

```julia
T = Float32
lstm = LSTM(T, 100, 100)
# one-step
c = Var(zeros(T,100))
h = Var(zeros(T,100))
c, h = lstm(x, c, h)
# n-step
h = lstm(x)
```
"""
type LSTM
    w::Var
    b::Var
end

function LSTM{T}(::Type{T}, insize::Int, hsize::Int)
    w = uniform(T, -0.001, 0.001, hsize*4, insize)
    u = orthogonal(T, hsize*4, hsize)
    w = cat(2, w, u)
    b = zeros(T, size(w,1))
    b[1:hsize] = ones(T, hsize) # forget gate initializes to 1
    LSTM(zerograd(w), zerograd(b))
end

function (f::LSTM)(x::Var, h::Var, c::Var)
    ys = Var[]
    for i = 1:size(x.data,2)
        h, c = onestep(f, x[:,i], h, c)
        push!(ys, h)
    end
    y = cat(2, ys...)
    y
end

function (f::LSTM)(x::Var)
    T = eltype(x.data)
    n = size(f.w.data,1) ÷ 4
    h = Var(zeros(T,n))
    c = Var(zeros(T,n))
    f(x, h, c)
end

function onestep(lstm::LSTM, x::Var, h::Var, c::Var)
    n = size(h.data, 1)
    a = lstm.w * cat(1,x,h) + lstm.b
    f = sigmoid(a[1:n])
    i = sigmoid(a[n+1:2n])
    o = sigmoid(a[2n+1:3n])
    c = f .* c + i .* tanh(a[3n+1:4n])
    h = o .* tanh(c)
    h, c
end

type BiLSTM
    fw
    bw
end

function BiLSTM{T}(::Type{T}, insize::Int, hsize::Int)
    BiLSTM(LSTM(T,insize,hsize), LSTM(T,insize,hsize))
end

function (f::BiLSTM)(x::Var)
end
