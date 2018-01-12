export LSTM, BiLSTM

struct LSTM
    hsize::Int
    nlayers::Int
    droprate::Float64
    rev::Bool
    w
    b
end

function LSTM(::Type{T}, hsize::Int, nlayers::Int, droprate::Float64;
    init_w=Orthogonal(), init_u=Orthogonal(), init_b=Fill(0), rev=false) where T

    ws = Var[]
    bs = Var[]
    for i = 1:nlayers
        wus = []
        for k = 1:4
            w = init_w(T, hsize, hsize)
            u = init_u(T, hsize, hsize)
            push!(wus, cat(1,w,u))
        end
        w = cat(2, wus...)
        push!(ws, zerograd(w))
        b = init_b(T, 4hsize)
        push!(bs, zerograd(b))
    end
    LSTM(hsize, nlayers, droprate, rev, ws, bs)
end

function (lstm::LSTM)(x::Var, batchdims::Vector{Int})
    if isa(x.data, Array)
        lstm_tstep(x, batchdims, lstm.w[1], lstm.b[1], lstm.rev)
    elseif isa(x.data, CuArray)
        CUDNN.lstm(lstm.hsize, lstm.nlayers, lstm.droprate, lstm.w, x.data, batchdims)
    end
end

# If mode in rnnDesc was set to CUDNN_LSTM values of 0, 1, 2 and 3 reference bias
# applied to the input from the previous layer, value of 4, 5, 6 and 7 reference bias
# applied to the recurrent input.
# ‣ Values 0 and 4 reference the input gate.
# ‣ Values 1 and 5 reference the forget gate.
# ‣ Values 2 and 6 reference the new memory gate.
# ‣ Values 3 and 7 reference the output gate.
function compile(lstm::LSTM, backend::Backend)
    h = lstm.hsize
    T = eltype(lstm.w[1])
    w = T[]
    append!(w, lstm.w[1].data[1:h,1:h])
    append!(w, lstm.w[1].data[1:h,h+1:2h])
    append!(w, lstm.w[1].data[1:h,2h+1:3h])
    append!(w, lstm.w[1].data[1:h,3h+1:4h])
    append!(w, lstm.w[1].data[h+1:2h,1:h])
    append!(w, lstm.w[1].data[h+1:2h,h+1:2h])
    append!(w, lstm.w[1].data[h+1:2h,2h+1:3h])
    append!(w, lstm.w[1].data[h+1:2h,3h+1:4h])
    append!(w, lstm.b[1].data)
    append!(w, zeros(lstm.b[1].data))
    w = compile(w, backend)
    LSTM(lstm.hsize, lstm.nlayers, lstm.droprate, lstm.rev, w, nothing)
end

function lstm_tstep(x::Var, batchdims::Vector{Int}, w::Var, b::Var, rev::Bool)
    @assert sum(batchdims) == size(x,2)
    @assert issorted(batchdims, rev=true)

    cumdims = Array{Int}(length(batchdims)+1)
    cumdims[1] = 1
    for i = 1:length(batchdims)
        cumdims[i+1] = cumdims[i] + batchdims[i]
    end

    #ht = concat(2, [h0 for i=1:length(batchdims)]...)
    #ct = concat(2, [c0 for i=1:length(batchdims)]...)
    hsize = size(w,2) ÷ 4
    h0 = Var(ones(eltype(x),hsize,length(batchdims)))
    c0 = Var(ones(eltype(x),hsize,length(batchdims)))
    ht = h0
    ct = c0
    hts = Var[]
    cts = Var[]
    for t = 1:batchdims[1]
        xts = Var[]
        for k = 1:length(batchdims)
            t > batchdims[k] && break
            i = cumdims[k]
            i += rev ? batchdims[k]-t : t-1
            push!(xts, x[:,i:i])
        end
        xt = concat(2, xts...)
        if size(ht,2) > size(xt,2)
            ht = ht[:,1:size(xt,2)]
            ct = ct[:,1:size(xt,2)]
        end
        ht, ct = lstm_onestep(w, b, xt, ht, ct)
        push!(hts, ht)
        push!(cts, ct)
        ht, ct = h0, c0
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

function catvec(arrays)
    T = eltype(arrays[1])
    n = sum(length, arrays)
    dest = similar(arrays[1], n)
    offset = 1
    for x in arrays
        copy!(dest, offset, x, 1, length(x))
        offset += length(x)
    end
    dest
end

function test_rnn()
    T = Float32
    x = CuArray(randn(T,10,12))
    batchdims = [5,4,3]

    insize = 10
    outsize = 10
    nlayers = 1
    W = cat(2, [randn(T,insize,outsize) for i=1:8]...)
    W = CuArray(W)
    #U = cat(2, [randn(T,outsize,outsize) for i=1:4]...)
    #W = cat(1, W, U) |> CuArray
    b = zeros(T, 8outsize) |> CuArray
    w = catvec([W,b])
    #h0 = zeros(T, outsize, 1)
    #c0 = zeros(T, outsize, 1)

    rnn = RNN(outsize, nlayers, 0.5, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD, w)
    y = rnn(xs, false)
    println(size(y))
    println(vec(y))
end
