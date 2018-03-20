export LSTM

mutable struct LSTM
    insize::Int
    hsize::Int
    nlayers::Int
    droprate::Float64
    bidirectional::Bool
    params::Tuple
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
    init_w=Normal(0,0.001), init_u=Orthogonal(), init_b=Fill(0), init_h0=Fill(0), init_c0=Fill(0)) where T

    ws = Var[]
    bs = Var[]
    coef = bidirectional ? 2 : 1
    for l = 1:nlayers
        for _ = 1:coef
            wus = []
            for i = 1:4
                s = l == 1 ? insize : hsize*coef
                w = init_w(T, s, hsize)
                u = init_u(T, hsize, hsize)
                push!(wus, cat(1,w,u))
            end
            w = cat(2, wus...)
            push!(ws, zerograd(w))
            b = init_b(T, 4hsize)
            push!(bs, zerograd(b))
        end
    end
    h0s = [Var(init_h0(T,hsize,1)) for i=1:nlayers*coef]
    c0s = [Var(init_c0(T,hsize,1)) for i=1:nlayers*coef]
    params = (ws,bs,h0s,c0s)
    LSTM(insize, hsize, nlayers, droprate, bidirectional, params)
end

function (lstm::LSTM)(x::Var, batchdims::Vector{Int})
    if isa(x.data, Array)
        ws,bs,h0s,c0s = lstm.params
        h = x
        coef = lstm.bidirectional ? 2 : 1
        for l = 1:lstm.nlayers
            i = (l-1) * coef + 1
            h1 = lstm_tstep(h, batchdims, ws[i], bs[i], h0s[i], c0s[i], false)
            if lstm.bidirectional
                h2 = lstm_tstep(h, batchdims, ws[i+1], bs[i+1], h0s[i+1], c0s[i+1], true)
                h = concat(1, h1, h2)
            else
                h = h1
            end
        end
        h
    elseif isa(x.data, CuArray)
        setcuda!(lstm)
        w, = lstm.params
        @assert issorted(batchdims, rev=true)
        t_x, t_batchdims = transpose_batch(x.data, batchdims)
        dir = lstm.bidirectional ? CUDNN.CUDNN_BIDIRECTIONAL : CUDNN.CUDNN_UNIDIRECTIONAL
        mode = CUDNN.CUDNN_LSTM
        t_y, work = rnn(lstm.insize, lstm.hsize, lstm.nlayers, lstm.droprate, dir, mode,
            w, t_x, batchdims, training=CONFIG.train)

        y, _ = transpose_batch(t_y, t_batchdims)
        Var(y, (lstm,x,batchdims,work,w))
    else
        throw("Invalid type.")
    end
end

function addgrad!(y::Var, lstm::LSTM, x::Var, batchdims::Vector{Int}, work, w::Var)
    t_gy, t_batchdims = transpose_batch(y.grad, batchdims)
    t_gx = CUDNN.backward_data(rnn, t_gy, work) # this call is required for backward_weights
    gx, _ = transpose_batch(t_gx, t_batchdims)
    isvoid(x.grad) || BLAS.axpy!(eltype(y)(1), gx, x.grad)
    CUDNN.backward_weights!(rnn, work)
end

function lstm_tstep(x::Var, batchdims::Vector{Int}, w::Var, b::Var, h0::Var, c0::Var, rev::Bool)
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
        ht, ct = lstm_onestep(w, b, xt, ht, ct)
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

function lstm_onestep(w, b, xt, ht, ct)
    a = linear(concat(1,xt,ht), w, b)
    n = size(a,1) ÷ 4
    i = sigmoid(a[1:n,:])
    f = sigmoid(a[n+1:2n,:])
    ct = f.*ct + i.*tanh(a[2n+1:3n,:])
    o = sigmoid(a[3n+1:4n,:])
    ht = o .* tanh(ct)
    ht, ct
end

# If mode in rnnDesc was set to CUDNN_LSTM values of 0, 1, 2 and 3 reference bias
# applied to the input from the previous layer, value of 4, 5, 6 and 7 reference bias
# applied to the recurrent input.
# ‣ Values 0 and 4 reference the input gate.
# ‣ Values 1 and 5 reference the forget gate.
# ‣ Values 2 and 6 reference the new memory gate.
# ‣ Values 3 and 7 reference the output gate.
function setcuda!(lstm::LSTM)
    isa(lstm.w.data,CuArray) && return
    param = eltype(lstm.ws[1])[]
    hx = eltype(lstm.ws[1])[]
    cx = eltype(lstm.ws[1])[]
    coef = lstm.bidirectional ? 2 : 1
    for l = 1:lstm.nlayers
        for d = 1:coef
            i = (l-1)*coef + d
            w = lstm.ws[i].data
            n = l == 1 ? lstm.insize : lstm.hsize*coef
            append!(param, w[1:n,:])
            append!(param, w[n+1:end,:])
            append!(hx, lstm.h0s[i].data)
            append!(cx, lstm.c0s[i].data)
        end
    end
    for l = 1:lstm.nlayers
        for d = 1:coef
            i = (l-1)*coef + d
            b = lstm.bs[i].data
            append!(param, b)
            append!(param, zeros(b)) # CUDNN requires bias for U
        end
    end
    w = CuArray(param)
    lstm.params = (w,)
end
