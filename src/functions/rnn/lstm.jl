export LSTM

"""
    LSTM(T::Type, xsize::Int, hsize::Int)

Long short-term memory (LSTM).

```julia
T = Float32
lstm = LSTM(T, 100, 100)
h = lstm(x)
```
"""
type LSTM
    w::Var
    b::Var
    h0::Var
    c0::Var
end

function LSTM{T}(::Type{T}, insize::Int, outsize::Int)
    w = rand(T, 4outsize, insize)
    w = w * T(0.002) - T(0.001)
    u = orthogonal(T, 4outsize, outsize)
    w = cat(2, w, u)
    b = zeros(T, size(w,1))
    b[1:outsize] = ones(T, outsize) # forget gate initializes to 1
    h0 = zeros(T, outsize)
    c0 = zeros(T, outsize)
    LSTM(zerograd(w), zerograd(b), zerograd(h0), zerograd(c0))
end

function (lstm::LSTM)(x::Var, h::Var, c::Var)
    a = lstm.w * cat(1,x,h) + lstm.b
    n = size(h.data, 1)
    f = sigmoid(a[1:n])
    i = sigmoid(a[n+1:2n])
    o = sigmoid(a[2n+1:3n])
    c = f .* c + i .* tanh(a[3n+1:4n])
    h = o .* tanh(c)
    h, c
end

function (lstm::LSTM)(x::Var)
    h, c = lstm.h0, lstm.c0
    n = size(x.data, 1)
    hs = Var[]
    for i = 1:size(x.data,2)
        h, c = lstm(x[(i-1)*n+1:i*n], h, c)
        push!(hs, h)
    end
    cat(2, hs...)
end

function lstm_fast{T}(out::Var, x::Matrix{T}, h::Vector{T}, c::Vector{T})
    fio = lstm.w * cat(1,x,h) + lstm.b
    n = size(h.data, 1)
    @inbounds for i = 1:3n
        a[i] = sigmoid(a[i])
    end
    @inbounds for i = 3n+1:4n
        a[i] = tanh(a[i])
    end
    for i = 1:n
        cc[i] = fio[i] * c[i] + i[i+n] * tanh(fio)
    end
end

#=
"""
Batched LSTM
"""
function (f::LSTM)(xs::Vector{Var}, h::Var, c::Var; rev=false)
    y = Var(nothing, f, (xs,h,c))

    rev && (xs = reverse(xs))
    ys = Array{Var}(length(xs))
    for i = 1:length(xs)
        h, c = f(xs[i], h, c)
        ys[i] = h
    end
    rev && (ys = reverse(ys))
    cat(2, ys)
end

function (f::LSTM)(x::Var, h::Var, c::Var; rev=false)
    ys = Var[]
    if rev == false
        for i = 1:size(x.data,2)
            h, c = onestep(f, x[:,i], h, c)
            push!(ys, h)
        end
    else
        for i = size(x.data,2):-1:1
            h, c = onestep(f, x[:,i], h, c)
            push!(ys, h)
        end
        ys = reverse(ys)
    end
    cat(2, ys...)
end

function (f::LSTM)(x::Var; rev=false)
    T = eltype(x.data)
    n = size(f.w.data,1) ÷ 4
    h = Var(zeros(T,n))
    c = Var(zeros(T,n))
    f(x, h, c, rev=rev)
end
=#

#=
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
=#
