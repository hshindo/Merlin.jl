export LSTM

mutable struct LSTM <: Functor
    insize::Int
    hsize::Int
    nlayers::Int
    droprate::Float64
    bidirectional::Bool
    params::Vector      # W, U, b, h, c
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
            Us = []
            for i = 1:4
                s = l == 1 ? insize : hsize*coef
                push!(Ws, init_W(T,s,hsize))
                push!(Us, init_U(T,hsize,hsize))
            end
            W = cat(2, Ws...)
            U = cat(2, Us...)
            b = init_b(T, 4hsize)
            h = init_h(T, hsize)
            c = init_c(T, hsize)
            push!(params, (param(W),param(U),param(b),param(h),param(c)))
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

function lstm_tstep(x::Var, batchdims::Vector{Int}, W::Var, U::Var, b::Var, h0::Var, c0::Var, rev::Bool)
    @assert sum(batchdims) == size(x,2)
    WU = concat(1, W, U)

    cumdims = Array{Int}(length(batchdims)+1)
    cumdims[1] = 1
    for i = 1:length(batchdims)
        cumdims[i+1] = cumdims[i] + batchdims[i]
    end
    perm = sortperm(batchdims, rev=true)

    hsize = length(h0)
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
        ht, ct = lstm_onestep(xt, WU, b, ht, ct)
        for j = 1:length(perm)
            p = perm[j]
            t > batchdims[p] && break
            i = cumdims[p]
            i += rev ? batchdims[p]-t : t-1
            hts[i] = ht[:,j:j]
            cts[i] = ct[:,j:j]
        end
    end
    concat(2, hts...)
end

function lstm_onestep(xt::Var, WU::Var, b::Var, ht::Var, ct::Var)
    a = linear(concat(1,xt,ht), WU, b)
    n = size(a,1) ÷ 4
    i = sigmoid(a[1:n,:])
    f = sigmoid(a[n+1:2n,:])
    ct = f.*ct + i.*tanh(a[2n+1:3n,:])
    o = sigmoid(a[3n+1:4n,:])
    ht = o .* tanh(ct)
    ht, ct
end

function batchsort(x::UniArray, batchdims::Vector{Int})
    perm = sortperm(batchdims, rev=true)
    cumdims = cumsum(batchdims)
    front = Base.front(size(x))
    view(x)
end

function lstm_cudnn(lstm::LSTM, x::Var, batchdims::Vector{Int})
    xs = unsafe_split(x.data, batchdims)
    perm = sortperm(batchdims, rev=true)
    xs = xs[perm]
    x = cat(ndims(x), xs...)

    Ws = Var[]
    h0s = Var[]
    c0s = Var[]
    for (W,U,b,h,c) in lstm.params
        push!(Ws, vec(W), vec(U))
    end
    for (W,U,b,h,c) in lstm.params
        push!(Ws, b, b)
        push!(h0s, h)
        push!(c0s, c)
    end
    W = concat(1, Ws...)
    h0 = concat(1, h0s...)
    c0 = concat(1, c0s...)
    t_x, t_batchdims = transpose_batch(x.data, batchdims)

    dir = lstm.bidirectional ? CUDNN.CUDNN_BIDIRECTIONAL : CUDNN.CUDNN_UNIDIRECTIONAL
    mode = CUDNN.CUDNN_LSTM
    t_y, work = CUDNN.rnn(lstm.insize, lstm.hsize, lstm.nlayers, lstm.droprate, dir, mode,
        W.data, t_x, t_batchdims, istrain())
    y, _ = transpose_batch(t_y, t_batchdims)

    ys = split(y, batchdims[perm])
    y = cat(ndims(y), ys[perm]...)
    Var(y, (lstm,x,batchdims,work,W))
end

function transpose_dims(dims::Vector{Int})
    @assert issorted(dims, rev=true)
    k = length(dims)
    t_dims = Int[]
    for t = 1:dims[1]
        while dims[k] < t
            k -= 1
        end
        push!(t_dims, k)
    end
    t_dims
end

function cumsum_cint(dims::Vector{Int})
    cumdims = Array{Cint}(length(dims)+1)
    cumdims[1] = 0
    for i = 2:length(cumdims)
        cumdims[i] = cumdims[i-1] + dims[i-1]
    end
    cumdims
end

@generated function transpose_batch(x::CuMatrix{T}, batchdims_x::Vector{Int}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void transpose_batch(int n, $Ct *y, int *cumdimsY, $Ct *x, int *cumdimsX) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n*cumdimsY[1]*cumdimsX[1]) return;

        int vj = idx / n;
        int vi = idx - vj * n;
        int j = vj / cumdimsY[1];
        int i = vj - j * cumdimsY[1];
        if (cumdimsY[j] + i < cumdimsY[j+1]) {
            int idxY = (cumdimsY[j] + i) * n + vi;
            int idxX = (cumdimsX[i] + j) * n + vi;
            y[idxY] = x[idxX];
        }
    }""")
    quote
        batchdims_y = transpose_dims(batchdims_x)
        cumdims_x = CuArray(cumsum_cint(batchdims_x))
        cumdims_y = CuArray(cumsum_cint(batchdims_y))

        y = similar(x)
        gdims, bdims = cudims(size(x,1)*batchdims_y[1]*batchdims_x[1])
        $k(gdims, bdims, size(x,1), pointer(y), pointer(cumdims_y), pointer(x), pointer(cumdims_x))
        y, batchdims_y
    end
end

function addgrad!(y::Var, lstm::LSTM, x::Var, batchdims::Vector{Int}, work, w::Var)
    t_gy, t_batchdims = transpose_batch(y.grad, batchdims)
    t_gx = CUDNN.∇rnn_data(work, t_gy) # this call is required for ∇rnn_weights!
    gx, _ = transpose_batch(t_gx, t_batchdims)
    isvoid(x.grad) || add!(x.grad, gx)
    isvoid(w.grad) || CUDNN.∇rnn_weights!(work, w.grad)
end
