export LSTM

mutable struct LSTM
    insize::Int
    hsize::Int
    nlayers::Int
    droprate::Float64
    bidirectional::Bool
    params::Vector          # W, U, b, h, c
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
            W = param(cat(Ws...,dims=2))
            U = param(cat(Us...,dims=2))
            b = param(init_b(T,4hsize))
            h = param(init_h(T,hsize))
            c = param(init_c(T,hsize))
            push!(params, (W,U,b,h,c))
        end
    end
    LSTM(insize, hsize, nlayers, droprate, bidirectional, params)
end

function (lstm::LSTM)(x::Var, batchdims::Vector{Int})
    configure!(getparams(lstm)...)
    if iscpu()
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
    elseif iscuda()
        lstm_cudnn(lstm, xs)
    else
        throw("Invalid backend.")
    end
end
(lstm::LSTM)(x::Node, batchdims) = Node(lstm, x, batchdims)

function lstm_tstep(x::Var, batchsize::Vector{Int}, W::Var, U::Var, b::Var, h0::Var, c0::Var, rev::Bool)
    @assert sum(batchsize) == size(x,2)
    WU = concat(1, W, U)

    cumdims = Array{Int}(undef, length(batchsize)+1)
    cumdims[1] = 1
    for i = 1:length(batchsize)
        cumdims[i+1] = cumdims[i] + batchsize[i]
    end
    perm = sortperm(batchsize, rev=true)

    hsize = length(h0)
    ht = concat(2, [h0 for i=1:length(batchsize)]...)
    ct = concat(2, [c0 for i=1:length(batchsize)]...)
    hts = Array{Var}(undef, size(x,2))
    cts = Array{Var}(undef, size(x,2))
    for t = 1:batchsize[perm[1]]
        xts = Var[]
        for p in perm
            t > batchsize[p] && break
            i = cumdims[p]
            i += rev ? batchsize[p]-t : t-1
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
            t > batchsize[p] && break
            i = cumdims[p]
            i += rev ? batchsize[p]-t : t-1
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
    ct = dot(f,ct) + dot(i,tanh(a[2n+1:3n,:]))
    o = sigmoid(a[3n+1:4n,:])
    ht = dot(o, tanh(ct))
    ht, ct
end

function batchsort(x::UniArray, batchsize::Vector{Int})
    perm = sortperm(batchsize, rev=true)
    cumdims = cumsum(batchsize)
    front = Base.front(size(x))
    view(x)
end

function lstm_cudnn(lstm::LSTM, x::Var, batchsize::Var)
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

    xs = unsafe_split(x.data, batchsize)
    perm = sortperm(batchsize, rev=true)
    t_x, t_batchsize = transpose_batch(xs[perm])

    dir = lstm.bidirectional ? CUDNN.CUDNN_BIDIRECTIONAL : CUDNN.CUDNN_UNIDIRECTIONAL
    mode = CUDNN.CUDNN_LSTM
    t_y, work = CUDNN.rnn(lstm.insize, lstm.hsize, lstm.nlayers, lstm.droprate, dir, mode,
        W.data, t_x, t_batchsize, istrain())

    y, _ = transpose_batch(unsafe_split(t_y,t_batchsize))
    ys = unsafe_split(y, batchsize[perm])
    y = cat(ys[perm]...,dims=ndims(y))
    Var(y, (lstm,x,batchsize,work,W))
end

function addgrad!(y::Var, lstm::LSTM, x::Var, batchsize::Vector{Int}, work, w::Var)
    t_gy, t_batchsize = transpose_batch(y.grad, batchsize)
    t_gx = CUDNN.∇rnn_data(work, t_gy) # this call is required for ∇rnn_weights!
    gx, _ = transpose_batch(t_gx, t_batchsize)
    isvoid(x.grad) || addto!(x.grad, gx)
    isvoid(w.grad) || CUDNN.∇rnn_weights!(work, w.grad)
end


@generated function transpose_batch(xs::Vector{CuMatrix{T}}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void transpose_batch(int n, $Ct *y, int *cumdimsY, $Ct **xs, int *cumdimsX) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n*cumdimsY[1]*cumdimsX[1]) return;

        int vj = idx / n;
        int vi = idx - vj * n;
        int j = vj / cumdimsY[1];
        int i = vj - j * cumdimsY[1];
        if (cumdimsY[j] + i < cumdimsY[j+1]) {
            int idxY = (cumdimsY[j] + i) * n + vi;
            int idxX = j * n + vi;
            y[idxY] = xs[i][idxX];
        }
    }""")
    quote
        batchsize_x = Array{Int}(length(xs))
        for i = 1:length(xs)
            batchsize_x[i] = size(xs[i], 2)
        end
        batchsize_y = transpose_dims(batchsize_x)
        cumdims_x = CuArray(cumsum_cint(batchsize_x))
        cumdims_y = CuArray(cumsum_cint(batchsize_y))
        y = CuArray{T}(size(xs[1],1), sum(batchsize_x))
        p_xs = CuArray(map(pointer,xs))
        gdims, bdims = cudims(size(xs[1],1)*batchsize_y[1]*batchsize_x[1])
        $k(gdims, bdims, size(xs[1],1), pointer(y), pointer(cumdims_y), pointer(p_xs), pointer(cumdims_x))
        y, batchsize_y
    end
end

@generated function transpose_batch2(x::CuMatrix{T}, batchsize_x::Vector{Int}) where T
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
        batchsize_y = transpose_dims(batchsize_x)
        cumdims_x = CuArray(cumsum_cint(batchsize_x))
        cumdims_y = CuArray(cumsum_cint(batchsize_y))

        y = similar(x)
        gdims, bdims = cudims(size(x,1)*batchsize_y[1]*batchsize_x[1])
        $k(gdims, bdims, size(x,1), pointer(y), pointer(cumdims_y), pointer(x), pointer(cumdims_x))
        y, batchsize_y
    end
end
