export LSTM

"""
    LSTM(T::Type, size::Int)

Long short-term memory (LSTM).

```julia
T = Float32
lstm = LSTM(T, 100, ()->rand()*0.02-0.01)
# one-step
c = Var(zeros(T,n))
h = Var(zeros(T,n))
c, h = lstm(x, c, h)
# n-step
h = lstm(x)
```
"""
type LSTM
    W
    b
end

function LSTM{T}(::Type{T}, size::Int, init::Function)
    w = reshape(T[init() for i=1:size*4*size], size*4, size)
    u = orthogonal(T, size*4, size)
    w = zerograd(cat(2,w,u))
    b = zerograd(zeros(T,size*4))
    b.data[1:size] = ones(T, size) # forget gate initializes to 1
    LSTM(w, b)
end

function (lstm::LSTM)(x::Var, c=nothing, h=nothing)
    T = eltype(lstm.b.data)
    n = Int(length(lstm.b.data)/4)
    size(x.data,1) == n || throw("Length of x is invalid.")
    isvoid(c) && (c = Var(zeros(T,n)))
    isvoid(h) && (h = Var(zeros(T,n)))

    ndims(x.data) == 1 && return onestep(lstm, x, c, h)
    hs = Var[]
    for i = 1:size(x.data,2)
        c, h = onestep(lstm, x[:,i], c, h)
        push!(hs, h)
    end
    cat(2, hs...)
end

function onestep(lstm::LSTM, x::Var, c::Var, h::Var)
    n = size(x.data, 1)
    a = lstm.W * cat(1,x,h) + lstm.b
    f = sigmoid(a[1:n])
    i = sigmoid(a[n+1:2n])
    o = sigmoid(a[2n+1:3n])
    c = f .* c + i .* tanh(a[3n+1:4n])
    h = o .* tanh(c)
    c, h
end
