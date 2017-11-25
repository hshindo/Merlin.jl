export LSTM

doc"""
    LSTM(::Type{T}, insize::Int, outsize::Int, [init_W=Uniform(0.001), init_U=Orthogonal()])

Long Short-Term Memory network.

```math
\begin{align*}
\mathbf{f}_{t} & =\sigma_{g}(W_{f}\mathbf{x}_{t}+U_{f}\mathbf{h}_{t-1}+\mathbf{b}_{f})\\
\mathbf{i}_{t} & =\sigma_{g}(W_{i}\mathbf{x}_{t}+U_{i}\mathbf{h}_{t-1}+\mathbf{b}_{i})\\
\mathbf{o}_{t} & =\sigma_{g}(W_{o}\mathbf{x}_{t}+U_{o}\mathbf{h}_{t-1}+\mathbf{b}_{o})\\
\mathbf{c}_{t} & =\mathbf{f}_{t}\odot\mathbf{c}_{t-1}+\mathbf{i}_{t}\odot\sigma_{c}(W_{c}\mathbf{x}_{t}+U_{c}\mathbf{h}_{t-1}+\mathbf{b}_{c})\\
\mathbf{h}_{t} & =\mathbf{o}_{t}\odot\sigma_{h}(\mathbf{c}_{t})
\end{align*}
```

* ``x_t \in R^{d}``: input vector to the LSTM block
* ``f_t \in R^{h}``: forget gate's activation vector
* ``i_t \in R^{h}``: input gate's activation vector
* ``o_t \in R^{h}``: output gate's activation vector
* ``h_t \in R^{h}``: output vector of the LSTM block
* ``c_t \in R^{h}``: cell state vector
* ``W \in R^{h \times d}``, ``U \in R^{h \times h}`` and ``b \in R^{h}``: weight matrices and bias vectors
* ``\sigma_g``: sigmoid function
* ``\sigma_c``: hyperbolic tangent function
* ``\sigma_h``: hyperbolic tangent function

# 👉 Example
```julia
T = Float32
f = LSTM(T, 100, 100)
h = f(x)
```
"""
struct LSTM
    WU::Var
    b::Var
    h0::Var
    c0::Var
end

function LSTM(::Type{T}, insize::Int, outsize::Int; init_W=Uniform(0.001), init_U=Orthogonal()) where T
    W = random(init_W, T, 4outsize, insize)
    U = random(init_U, T, 4outsize, insize)
    wu = cat(2, w, u)
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
