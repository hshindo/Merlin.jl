export recurrent
export LSTM, BiLSTM

function recurrent(f, x::Var, batchdims1::Var, h0::Var; rev=false)
    batchdims = batchdims1.data
    @assert sum(batchdims) == size(x.data,2)
    cumdims = Array{Int}(length(batchdims)+1)
    cumdims[1] = 1
    for i = 1:length(batchdims)
        cumdims[i+1] = cumdims[i] + batchdims[i]
    end
    perm = sortperm(batchdims, rev=true)
    h = h0
    hs = Array{Var}(size(x,2))
    for t = 1:batchdims[perm[1]]
        xts = Var[]
        for p in perm
            t > batchdims[p] && break
            i = cumdims[p]
            i += rev ? batchdims[p]-t : t-1
            push!(xts, x[:,i:i])
        end
        xt = concat(2, xts...)
        if size(h,2) < size(xt,2)
            @assert size(h,2) == 1
            h = concat(2, ntuple(_ -> h, size(xt,2))...)
        elseif size(h,2) > size(xt,2)
            h = h[:,1:size(xt,2)]
        end
        xt = concat(1, xt, h)
        h = f(xt)
        for j = 1:length(perm)
            p = perm[j]
            t > batchdims[p] && break
            i = cumdims[p]
            i += rev ? batchdims[p]-t : t-1
            hs[i] = h[:,j:j]
        end
    end
    concat(2, hs...)
end

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
x = Var(rand(T,100,10))
f = LSTM(T, 100, 100)
h = f(x)
```
"""
mutable struct LSTM
    WU::Var
    b::Var
    h0::Var
    c0::Var
end

function LSTM(::Type{T}, insize::Int, outsize::Int; init_W=Xavier(), init_U=Orthogonal()) where T
    W = init_W(T, insize, 4outsize)
    U = init_U(T, insize, 4outsize)
    WU = cat(1, W, U)
    b = zeros(T, 4outsize)
    b[1:outsize] = ones(T, outsize) # forget gate initializes to 1
    h0 = zeros(T, outsize, 1)
    c0 = zeros(T, outsize, 1)
    LSTM(zerograd(WU), zerograd(b), zerograd(h0), zerograd(c0))
end

function (lstm::LSTM)(x::Var, batchdims; rev=false)
    isvoid(x.data) && return Var(nothing,(lstm,x,batchdims,(:rev,rev)))
    c = lstm.c0
    h = recurrent(x, batchdims, lstm.h0, rev=rev) do xt
        a = linear(xt, lstm.WU, lstm.b)
        n = size(a,1) ÷ 4
        f = sigmoid(a[1:n,:])
        i = sigmoid(a[n+1:2n,:])
        o = sigmoid(a[2n+1:3n,:])
        if size(c,2) < size(xt,2)
            @assert size(c,2) == 1
            c = concat(2, ntuple(_ -> c, size(xt,2))...)
        elseif size(c,2) > size(xt,2)
            c = c[:,1:size(xt,2)]
        end
        c = f .* c + i .* tanh(a[3n+1:4n,:])
        h = o .* tanh(c)
        h
    end
    h
end

doc"""
    BiLSTM(::Type{T}, insize::Int, outsize::Int, [init_W=Uniform(0.001), init_U=Orthogonal()])

Bi-directional Long Short-Term Memory network.
See `LSTM` for more details.
"""
mutable struct BiLSTM
    fwd::LSTM
    bwd::LSTM
end

function BiLSTM(::Type{T}, insize::Int, outsize::Int; init_W=Uniform(0.001), init_U=Orthogonal()) where T
    fwd = LSTM(T, insize, outsize, init_W=init_W, init_U=init_U)
    bwd = LSTM(T, insize, outsize, init_W=init_W, init_U=init_U)
    BiLSTM(fwd, bwd)
end

function (bilstm::BiLSTM)(x::Var, batchdims)
    isvoid(x.data) && return Var(nothing,(bilstm,x,batchdims))
    h1 = bilstm.fwd(x, batchdims)
    h2 = bilstm.bwd(x, batchdims, rev=true)
    concat(1, h1, h2)
end
