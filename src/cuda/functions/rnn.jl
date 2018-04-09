function lstm_cudnn(lstm::LSTM, xs::Vector{Var})
    Ws = Var[]
    h0s = Var[]
    c0s = Var[]
    coef = lstm.bidirectional ? 2 : 1
    for l = 1:lstm.nlayers
        for d = 1:coef
            i = (l-1)*coef + d
            p = lstm.params[i]
            n = l == 1 ? lstm.insize : lstm.hsize*coef
            push!(Ws, vec(p.W[1:n,:]), vec(p.W[n+1:end,:]))
        end
    end
    for p in lstm.params
        push!(Ws, p.b)
        push!(Ws, p.b) # CUDNN requires bias for U
        push!(h0s, p.h0)
        push!(c0s, p.c0)
    end
    W = concat(1, Ws...)
    h0 = concat(1, h0s...)
    c0 = concat(1, c0s...)

    batchdims = map(x -> size(x,2), xs)
    perm = sortperm(batchdims, rev=true)
    x = cat(2, map(x -> x.data, xs[perm])...)
    t_x, t_batchdims = batch_rnn(x, batchdims)

    dir = lstm.bidirectional ? CUDNN.CUDNN_BIDIRECTIONAL : CUDNN.CUDNN_UNIDIRECTIONAL
    mode = CUDNN.CUDNN_LSTM
    t_y, work = CUDNN.rnn(lstm.insize, lstm.hsize, lstm.nlayers, lstm.droprate, dir, mode,
        W.data, t_x, t_batchdims, training=CONFIG.train)

    y, _ = batch_rnn(t_y, t_batchdims)
    Var(y, (lstm,x,batchdims,work))
end

@generated function batch_rnn(x::CuMatrix{T}, batchdims_x::Vector{Int}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void batch_rnn(int vecsize, $Ct *y, int *cumdimsY, int sizeY, $Ct *x, int *cumdimsX, int sizeX) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= vecsize*sizeX*sizeY) return;

        int vi = idx / vecsize;
        int voffset = idx - vi * vecsize;
        int j = vi / sizeY;
        int i = vi - j * sizeY;
        if (cumdimsY[j] + i < cumdimsY[j+1]) {
            int idxY = (cumdimsY[j] + i) * vecsize + voffset;
            int idxX = (cumdimsX[i] + j) * vecsize + voffset;
            y[idxY] = x[idxX];
        }
    }""")
    quote
        cumdims_x = Array{Cint}(length(batchdims_x)+1)
        cumdims_x[1] = 0
        for i = 2:length(cumdims_x)
            cumdims_x[i] = cumdims_x[i-1] + batchdims_x[i-1]
        end

        batchdims_y = transpose_batchdims(batchdims_x)
        cumdims_y = Array{Cint}(length(batchdims_y)+1)
        cumdims_y[1] = 0
        for i = 2:length(cumdims_y)
            cumdims_y[i] = cumdims_y[i-1] + batchdims_y[i-1]
        end

        y = similar(x)
        gdims, bdims = cudims(size(x,1)*batchdims_y[1]*batchdims_x[1])
        culaunch($f, gdims, bdims, Cint(size(x,1)), Ptr{T}(y), Ptr{Cint}(CuArray(cumdims_y)), Cint(batchdims_y[1]),
            Ptr{T}(x), Ptr{Cint}(CuArray(cumdims_x)), Cint(batchdims_x[1]))
        y, batchdims_y
    end
end

function addgrad!(y::Var, lstm::LSTM, x::Var, batchdims::Vector{Int}, work, w::Var)
    t_gy, t_batchdims = batch_rnn(y.grad, batchdims)
    t_gx = CUDNN.backward_data(rnn, t_gy, work) # this call is required for backward_weights
    gx, _ = transpose_batch(t_gx, t_batchdims)
    isvoid(x.grad) || BLAS.axpy!(eltype(y)(1), gx, x.grad)
    CUDNN.backward_weights!(rnn, work)
end
