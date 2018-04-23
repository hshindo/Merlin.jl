export LSTM

mutable struct LSTM <: Functor
    insize::Int
    hsize::Int
    nlayers::Int
    droprate::Float64
    bidirectional::Bool
    params::Vector # w, b, h, c
end

function getparams(lstm::LSTM)
    params = Var[]
    for p in lstm.params
        append!(params, p)
    end
    params
end

doc"""
    LSTM(::Type{T}, insize::Int, hsize::Int, nlayers::Int, droprate::Float64, bidirectional::Bool,
        [init_w=Orthogonal(), init_u=Orthogonal(), init_b=Fill(0)])

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

```julia
T = Float32
x = Var(rand(T,100,10))
f = LSTM(T, 100, 100)
h = f(x)
```
"""
function LSTM(::Type{T}, insize::Int, hsize::Int, nlayers::Int, droprate::Float64, bidirectional::Bool;
    init_W=Normal(0,0.001), init_U=Orthogonal(), init_b=Fill(0), init_h=Fill(0), init_c=Fill(0)) where T

    params = []
    coef = bidirectional ? 2 : 1
    for l = 1:nlayers
        for _ = 1:coef
            Ws = []
            for i = 1:4
                s = l == 1 ? insize : hsize*coef
                W = init_W(T, s, hsize)
                U = init_U(T, hsize, hsize)
                push!(Ws, cat(1,W,U))
            end
            W = cat(2, Ws...)
            b = init_b(T, 4hsize)
            h = init_h(T, hsize)
            c = init_c(T, hsize)
            push!(params, (param(W),param(b),param(h),param(c)))
        end
    end
    LSTM(insize, hsize, nlayers, droprate, bidirectional, params)
end

function (lstm::LSTM)(x::Var, batchdims::Vector{Int})
    configure!(x, getparams(lstm)...)
    if iscpu()
        lstm_naive(lstm, x, batchdims)
    elseif iscuda()
        lstm_cudnn(lstm, x, batchdims)
    else
        throw("Invalid backend.")
    end
end

function lstm_naive(lstm::LSTM, x::Var, batchdims::Vector{Int})
    h = x
    coef = lstm.bidirectional ? 2 : 1
    for l = 1:lstm.nlayers
        i = (l-1) * coef + 1
        p = lstm.params[i]
        h1 = lstm_tstep(h, batchdims, p..., false)
        if lstm.bidirectional
            p = lstm.params[i+1]
            h2 = lstm_tstep(h, batchdims, p..., true)
            h = concat(1, h1, h2)
        else
            h = h1
        end
    end
    h
end

function lstm_tstep(x::Var, batchdims::Vector{Int}, W::Var, b::Var, h0::Var, c0::Var, rev::Bool)
    @assert sum(batchdims) == size(x,2)
    cumdims = Array{Int}(length(batchdims)+1)
    cumdims[1] = 1
    for i = 1:length(batchdims)
        cumdims[i+1] = cumdims[i] + batchdims[i]
    end
    perm = sortperm(batchdims, rev=true)

    hsize = size(W,2) ÷ 4
    ht = concat(2, [h0 for i=1:length(batchdims)]...)
    ct = concat(2, [c0 for i=1:length(batchdims)]...)
    hts = Array{Var}(size(x,2))
    cts = Array{Var}(size(x,2))
    for t = 1:batchdims[perm[1]]
        xts = Var[]
        for p in perm
            t > batchdims[p] && break
            i = cumdims[p]
            i += rev ? batchdims[p]-t : t-1
            push!(xts, x[:,i:i])
        end
        xt = concat(2, xts...)
        if size(ht,2) > size(xt,2)
            ht = ht[:,1:size(xt,2)]
            ct = ct[:,1:size(xt,2)]
        end
        ht, ct = lstm_onestep(xt, W, b, ht, ct)
        for j = 1:length(perm)
            p = perm[j]
            t > batchdims[p] && break
            i = cumdims[p]
            i += rev ? batchdims[p]-t : t-1
            hts[i] = ht[:,j:j]
            cts[i] = ct[:,j:j]
        end
    end
    h = concat(2, hts...)
    h
end

function lstm_onestep(xt::Var, W::Var, b::Var, ht::Var, ct::Var)
    a = linear(concat(1,xt,ht), W, b)
    n = size(a,1) ÷ 4
    i = sigmoid(a[1:n,:])
    f = sigmoid(a[n+1:2n,:])
    ct = f.*ct + i.*tanh(a[2n+1:3n,:])
    o = sigmoid(a[3n+1:4n,:])
    ht = o .* tanh(ct)
    ht, ct
end

#=
function lstm2(lstm::LSTM, xs::Vector{Var})
    configure!(xs)
    configure!(getparams(lstm))
    if iscpu()
        batchdims = map(x -> size(x,2), xs)
        h = concat(2, xs...)
        coef = lstm.bidirectional ? 2 : 1
        for l = 1:lstm.nlayers
            i = (l-1) * coef + 1
            p = lstm.params[i]
            h1 = lstm_tstep2(h, batchdims, p.W, p.b, p.h0, p.c0, false)
            if lstm.bidirectional
                p = lstm.params[i+1]
                h2 = lstm_tstep2(h, batchdims, p.W, p.b, p.h0, p.c0, true)
                h = concat(1, h1, h2)
            else
                h = h1
            end
        end
        h
    elseif iscuda()
        lstm_cudnn(lstm, xs)
    else
        throw("Invalid backend.")
    end
end

function lstm_tstep2(x::Var, batchdims::Vector{Int}, w::Var, b::Var, h0::Var, c0::Var, rev::Bool)
    @assert sum(batchdims) == size(x,2)
    cumdims = Array{Int}(length(batchdims)+1)
    cumdims[1] = 1
    for i = 1:length(batchdims)
        cumdims[i+1] = cumdims[i] + batchdims[i]
    end
    perm = sortperm(batchdims, rev=true)

    hsize = size(w,2) ÷ 4
    ht = concat(2, [h0 for i=1:length(batchdims)]...)
    ct = concat(2, [c0 for i=1:length(batchdims)]...)
    hts = Array{Var}(size(x,2))
    cts = Array{Var}(size(x,2))
    for t = 1:batchdims[perm[1]]
        xts = Var[]
        for p in perm
            t > batchdims[p] && break
            i = cumdims[p]
            i += rev ? batchdims[p]-t : t-1
            push!(xts, x[:,i:i])
        end
        xt = concat(2, xts...)
        if size(ht,2) > size(xt,2)
            ht = ht[:,1:size(xt,2)]
            ct = ct[:,1:size(xt,2)]
        end
        ht, ct = lstm_onestep(xt, w, b, ht, ct)
        for j = 1:length(perm)
            p = perm[j]
            t > batchdims[p] && break
            i = cumdims[p]
            i += rev ? batchdims[p]-t : t-1
            hts[i] = ht[:,j:j]
            cts[i] = ct[:,j:j]
        end
    end
    h = concat(2, hts...)
    h
end
=#
