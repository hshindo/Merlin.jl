export LSTM

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

function LSTM{T}(::Type{T}, xsize::Int, hsize::Int)
    w = uniform(T, -0.001, 0.001, hsize*4, xsize+hsize)
    u = orthogonal(T, hsize*4, xsize+hsize)
    w = cat(2, w, u)
    b = zeros(T,size(w,1))
    b[1:hsize] = ones(T, hsize) # forget gate initializes to 1
    LSTM(zerograd(w), zerograd(b))
end

function (lstm::LSTM)(x::Var, c=nothing, h=nothing)
    T = eltype(lstm.b.data)
    hsize = Int(length(lstm.b.data)/4)
    xsize = size(lstm.w.data,2) - hsize
    size(x.data,1) == xsize || throw("Length of x is invalid.")
    isvoid(c) && (c = Var(zeros(T,hsize)))
    isvoid(h) && (h = Var(zeros(T,hsize)))

    ndims(x.data) == 1 && return onestep(lstm, x, c, h)
    hs = Var[]
    for i = 1:size(x.data,2)
        c, h = onestep(lstm, x[:,i], c, h)
        push!(hs, h)
    end
    hs
end

function onestep(lstm::LSTM, x::Var, c::Var, h::Var)
    n = size(x.data, 1)
    a = lstm.w * cat(1,x,h) + lstm.b
    f = sigmoid(a[1:n])
    i = sigmoid(a[n+1:2n])
    o = sigmoid(a[2n+1:3n])
    c = f .* c + i .* tanh(a[3n+1:4n])
    h = o .* tanh(c)
    c, h
end
