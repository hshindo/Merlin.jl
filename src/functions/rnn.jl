export LSTM, BiLSTM

struct LSTM
    insize::Int
    hsize::Int
    nlayers::Int
    droprate::Float64
    ws::Vector{Var}
    bs::Vector{Var}
end

doc"""
    LSTM(::Type{T}, insize::Int, hsize::Int, nlayers::Int, droprate::Float64,
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

# 👉 Example
```julia
T = Float32
x = Var(rand(T,100,10))
f = LSTM(T, 100, 100)
h = f(x)
```
"""
function LSTM(::Type{T}, insize::Int, hsize::Int, nlayers::Int, droprate::Float64;
    init_w=Orthogonal(), init_u=Orthogonal(), init_b=Fill(0), init_h0=Fill(0), init_c0=Fill(0)) where T

    ws = Var[]
    bs = Var[]
    for l = 1:nlayers
        wus = []
        for i = 1:4
            s = l == 1 ? insize : hsize
            w = init_w(T, s, hsize)
            u = init_u(T, s, hsize)
            push!(wus, cat(1,w,u))
        end
        w = cat(2, wus...)
        push!(ws, zerograd(w))
        b = init_b(T, 4hsize)
        push!(bs, zerograd(b))
    end
    LSTM(insize, hsize, nlayers, droprate, ws, bs)
end

function (lstm::LSTM)(x::Var, batchdims::Vector{Int})
    lstm_tstep(x, batchdims, lstm.ws[1], lstm.bs[1], false)
end

function (rnn::CUDNN.RNN)(x::Var, batchdims::Vector{Int})
    t_x, t_batchdims = transpose_batch(x.data, batchdims)
    t_y, work = rnn(t_x, t_batchdims)
    y, _ = transpose_batch(t_y, t_batchdims)
    Var(y, (rnn,x,batchdims), work=work)
end

function addgrad!(y::Var, rnn::CUDNN.RNN, x::Var, batchdims::Vector{Int})
    t_gy, t_batchdims = transpose_batch(y.grad, batchdims)
    t_gx = CUDNN.backward_data(rnn, t_gy, y.work) # this call is required for backward_weights
    gx, _ = transpose_batch(t_gx, t_batchdims)
    isvoid(x.grad) || BLAS.axpy!(eltype(y)(1), gx, x.grad)
    CUDNN.backward_weights!(rnn, y.work)
end

function lstm_tstep(x::Var, batchdims::Vector{Int}, w::Var, b::Var, rev::Bool)
    @assert sum(batchdims) == size(x,2)
    cumdims = Array{Int}(length(batchdims)+1)
    cumdims[1] = 1
    for i = 1:length(batchdims)
        cumdims[i+1] = cumdims[i] + batchdims[i]
    end
    perm = sortperm(batchdims, rev=true)

    #ht = concat(2, [h0 for i=1:length(batchdims)]...)
    #ct = concat(2, [c0 for i=1:length(batchdims)]...)
    hsize = size(w,2) ÷ 4
    h0 = Var(zeros(eltype(x),hsize,length(batchdims)))
    c0 = Var(zeros(eltype(x),hsize,length(batchdims)))
    ht = h0
    ct = c0
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
function compile(lstm::LSTM, backend::CUDABackend)
    param = eltype(lstm.ws[1])[]
    for l = 1:lstm.nlayers
        w = lstm.ws[l].data
        n = size(w,1) ÷ 2
        append!(param, w[1:n,:])
        append!(param, w[n+1:2n,:])
        b = lstm.bs[l].data
        append!(param, b)
        append!(param, zeros(b)) # CUDNN requires bias for U
    end
    w = compile(param, backend)
    CUDNN.LSTM(lstm.insize, lstm.hsize, lstm.nlayers, lstm.droprate, w)
end
