export LSTM

mutable struct LSTM
    insize::Int
    hsize::Int
    nlayers::Int
    droprate::Float64
    bidir::Bool
    params
    iscuda::Bool
end

doc"""
    LSTM(::Type{T}, insize::Int, hsize::Int, nlayers::Int, droprate::Float64, bidir::Bool,
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
function LSTM(::Type{T}, insize::Int, hsize::Int, nlayers::Int, droprate::Float64, bidir::Bool;
    init_W=Normal(0,0.001), init_U=Orthogonal(), init_b=Fill(0), init_h=Fill(0), init_c=Fill(0)) where T

    params = []
    coef = bidir ? 2 : 1
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
    LSTM(insize, hsize, nlayers, droprate, bidir, params, false)
end

function configure!(lstm::LSTM)
    if iscuda() && !lstm.iscuda
        lstm.iscuda = true
        Ws = []
        hs = []
        cs = []
        for (W,U,b,h,c) in lstm.params
            push!(Ws, vec(W.data), vec(U.data))
        end
        for (W,U,b,h,c) in lstm.params
            push!(Ws, b.data, fill!(similar(b.data),0))
            push!(hs, h.data)
            push!(cs, c.data)
        end
        W = param(cat(Ws...,dims=1))
        h = param(cat(hs...,dims=1))
        c = param(cat(cs...,dims=1))
        configure!(W, h, c)
        lstm.params = (W,)
    end
end

function (lstm::LSTM)(x::Var, dims::Vector{Int})
    if iscuda()
        lstm.iscuda || configure!(lstm)
        configure!(x)
        W = lstm.params[1]
        t_xdata, t_dims = transpose_batch(x.data, dims)
        dir = lstm.bidir ? CUDNN.CUDNN_BIDIRECTIONAL : CUDNN.CUDNN_UNIDIRECTIONAL
        mode = CUDNN.CUDNN_LSTM
        t_ydata, work = CUDNN.rnn(lstm.insize, lstm.hsize, lstm.nlayers, lstm.droprate, dir, mode,
            W.data, t_xdata, t_dims, istrain())
        ydata, _ = transpose_batch(t_ydata, t_dims)
        Var(ydata, ∇lstm!, (lstm,x,dims,work,W))
    else
        h = x
        coef = lstm.bidir ? 2 : 1
        for l = 1:lstm.nlayers
            i = (l-1) * coef + 1
            p = lstm.params[i]
            h1 = lstm_tstep(h, dims, p..., false)
            if lstm.bidir
                p = lstm.params[i+1]
                h2 = lstm_tstep(h, dims, p..., true)
                h = concat(1, h1, h2)
            else
                h = h1
            end
        end
        h
    end
end
(lstm::LSTM)(x::Node, dims) = Node(lstm, (x,dims))

function lstm_tstep(x::Var, dims::Vector{Int}, W::Var, U::Var, b::Var, h::Var, c::Var, rev::Bool)
    @assert sum(dims) == size(x,2)
    @assert issorted(dims, rev=true)
    WU = concat(1, W, U)

    cumdims = Array{Int}(undef, length(dims)+1)
    cumdims[1] = 1
    for i = 1:length(dims)
        cumdims[i+1] = cumdims[i] + dims[i]
    end

    hsize = length(h)
    ht = concat(2, [h for i=1:length(dims)]...)
    ct = concat(2, [c for i=1:length(dims)]...)
    hts = Array{Var}(undef, size(x,2))
    cts = Array{Var}(undef, size(x,2))
    for t = 1:dims[1]
        xts = Var[]
        for j = 1:length(dims)
            d = dims[j]
            t > d && break
            k = cumdims[j]
            k += rev ? d-t : t-1
            push!(xts, x[:,k:k])
        end
        xt = concat(2, xts...)
        if size(ht,2) > size(xt,2)
            ht = ht[:,1:size(xt,2)]
            ct = ct[:,1:size(xt,2)]
        end
        ht, ct = lstm_onestep(xt, WU, b, ht, ct)
        for j = 1:length(dims)
            d = dims[j]
            t > d && break
            k = cumdims[j]
            k += rev ? d-t : t-1
            hts[k] = ht[:,j:j]
            cts[k] = ct[:,j:j]
        end
    end
    concat(2, hts...)
end

function lstm_onestep(xt::Var, WU::Var, b::Var, ht::Var, ct::Var)
    a = linear(concat(1,xt,ht), WU, b)
    n = size(a,1) ÷ 4
    i = sigmoid(a[1:n,:])
    f = sigmoid(a[n+1:2n,:])
    ct = (f .* ct) + (i .* tanh(a[2n+1:3n,:]))
    o = sigmoid(a[3n+1:4n,:])
    ht = o .* tanh(ct)
    ht, ct
end

function ∇lstm!(y::Var, f::LSTM, x::Var, dims, work, W)
    t_gy, t_dims = transpose_batch(y.grad, dims)
    t_gx = CUDNN.∇rnn_data(work, t_gy) # this call is required for ∇rnn_weights!
    gx, _ = transpose_batch(t_gx, t_dims)
    isnothing(x.grad) || addto!(x.grad, gx)
    isnothing(W.grad) || CUDNN.∇rnn_weights!(work, W.grad)
end

@generated function transpose_batch(x::CuMatrix{T}, dims::Vector{Int}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void transpose_batch(int n, $Ct *t_x, int *t_cumdims, $Ct *x, int *cumdims) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n*t_cumdims[1]*cumdims[1]) return;

        int vj = idx / n;
        int vi = idx - vj * n;
        int j = vj / t_cumdims[1];
        int i = vj - j * t_cumdims[1];
        if (t_cumdims[j] + i < t_cumdims[j+1]) {
            int t_idx = (t_cumdims[j] + i) * n + vi;
            idx = (cumdims[i] + j) * n + vi;
            t_x[t_idx] = x[idx];
        }
    }""")
    quote
        @assert issorted(dims, rev=true)
        t_x = similar(x)
        t_dims = transpose_dims(dims)
        cumdims = Array{Cint}(undef, length(dims)+1)
        t_cumdims = Array{Cint}(undef, length(t_dims)+1)
        cumdims[1] = 0
        t_cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + dims[i-1]
        end
        for i = 2:length(t_cumdims)
            t_cumdims[i] = t_cumdims[i-1] + t_dims[i-1]
        end
        cumdims = CuArray(cumdims)
        t_cumdims = CuArray(t_cumdims)

        gdims, bdims = cudims(size(x,1)*dims[1]*t_dims[1])
        $k(gdims, bdims, size(x,1), pointer(t_x), pointer(t_cumdims), pointer(x), pointer(cumdims))
        t_x, t_dims
    end
end

function transpose_dims(dims::Vector{Int})
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
