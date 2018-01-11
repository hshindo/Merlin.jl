# cudnnRNNMode_t
const CUDNN_RNN_RELU = Cint(0)
const CUDNN_RNN_TANH = Cint(1)
const CUDNN_LSTM = Cint(2)
const CUDNN_GRU = Cint(3)

# cudnnDirectionMode_t
const CUDNN_UNIDIRECTIONAL = Cint(0)
const CUDNN_BIDIRECTIONAL = Cint(1)

# cudnnRNNInputMode_t
const CUDNN_LINEAR_INPUT = Cint(0)
const CUDNN_SKIP_INPUT = Cint(1)

# cudnnRNNAlgo_t
const CUDNN_RNN_ALGO_STANDARD = Cint(0)
const CUDNN_RNN_ALGO_PERSIST_STATIC = Cint(1)
const CUDNN_RNN_ALGO_PERSIST_DYNAMIC = Cint(2)

mutable struct RNNDesc
    ptr::Cptr

    function RNNDesc(::Type{T}, hsize::Int, nlayers::Int, droprate::Float64, direction, mode, algo) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateRNNDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc, x -> @cudnn :cudnnDestroyRNNDescriptor (Cptr,) x)

        h = gethandle()
        dropdesc = DropoutDesc(droprate)
        @cudnn(:cudnnSetRNNDescriptor,
            (Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
            h, desc, hsize, nlayers, dropdesc, CUDNN_LINEAR_INPUT, direction, mode, algo, datatype(T))
        desc
    end
end

Base.unsafe_convert(::Type{Cptr}, desc::RNNDesc) = desc.ptr

### Size chart (Julia sizes for CUDNN calls)
# Note: For Julia calls, x and y do not need the initial 1 dimension and B,T are optional.
#
# x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
# xDesc: Array of T (1,X,B) descriptors
# y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
# yDesc: Array of T (1,Y,B) descriptors
# w: (1,1,W) where W = cudnnGetRNNParamsSize()
# hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
#
# Note: cudnn docs say min tensor dims 4 but RNN_example.cu uses 3D tensors

mutable struct RNN
    desc::RNNDesc
    hsize::Int
    nlayers::Int
    direction::Cint
    wdesc
    w
end

doc"""
    RNN

```julia
rnn = RNN(T, 10, 1, 0.5, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD)
```
"""
function RNN(hsize::Int, nlayers::Int, droprate::Float64, direction::Cint, mode::Cint, algo::Cint, w::CuVector{T}) where T
    rnndesc = RNNDesc(T, hsize, nlayers, droprate, direction, mode, algo)
    wdesc = FilterDesc(T, 1, 1, length(w))
    RNN(rnndesc, hsize, nlayers, direction, wdesc, w)
end

function (rnn::RNN)(xs::Vector{CuMatrix{T}}, inference=true) where T
    # x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
    # xDesc: Array of T (1,X,B) descriptors
    xdesc = map(xs) do x
        TensorDesc(T, 1, size(x)...)
    end
    x = catvec(xs)

    # hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
    coef = rnn.direction == CUDNN_UNIDIRECTIONAL ? 1 : 2
    hxdesc = TensorDesc(T, rnn.hsize, size(xs[1],2), rnn.nlayers*coef)

    # y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
    # yDesc: Array of T (1,Y,B) descriptors
    y = CuArray{T}(rnn.hsize*coef, sum(x -> size(x,2),xs))
    ydesc = map(xs) do x
        TensorDesc(T, 1, rnn.hsize*coef, size(x,2))
    end

    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNWorkspaceSize,
        (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
        h, rnn.desc, length(xdesc), xdesc, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    if inference
        @cudnn(:cudnnRNNForwardInference,
            (Cptr,Cptr,Cint,
            Ptr{Cptr},Cptr,     # x
            Cptr,Cptr,          # hx
            Cptr,Cptr,          # cx
            Cptr,Cptr,          # w
            Ptr{Cptr},Cptr,     # y
            Cptr,Cptr,          # hy
            Cptr,Cptr,          # cy
            Cptr,Csize_t),      # workspace
            h, rnn.desc, length(xdesc),
            xdesc, x,
            hxdesc, C_NULL,
            hxdesc, C_NULL,
            rnn.wdesc, rnn.w,
            ydesc, y,
            hxdesc, C_NULL,
            hxdesc, C_NULL,
            workspace, length(workspace))
    else
        ref = Ref{Csize_t}()
        @cudnn(:cudnnGetRNNTrainingReserveSize,
            (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
            h, rnn.desc, length(xdesc), xdesc, ref)
        reservespace = CuArray{UInt8}(Int(ref[]))

        @cudnn(:cudnnRNNForwardTraining,
            (Cptr,Cptr,Cint,
            Ptr{Cptr},Cptr,     # x
            Cptr,Cptr,          # hx
            Cptr,Cptr,          # cx
            Cptr,Cptr,          # w
            Ptr{Cptr},Cptr,     # y
            Cptr,Cptr,          # hy
            Cptr,Cptr,          # cy
            Cptr,Csize_t,       # workspace
            Cptr,Csize_t),      # reservespace
            h, rnn.desc, length(xdesc),
            xdesc, x,
            hxdesc, C_NULL,
            hxdesc, C_NULL,
            rnn.wdesc, rnn.w,
            ydesc, y,
            hxdesc, C_NULL,
            hxdesc, C_NULL,
            workspace, length(workspace),
            reservespace, length(workspace))
    end
    y
end

function getRNNParamSize(::Type{T}, desc, xdesc) where T
    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNParamsSize,
        (Cptr,Cptr,Cptr,Ptr{Csize_t},Cint),
        h, desc, xdesc, ref, datatype(T))
    println(Int(ref[]) ÷ sizeof(T))
end

function catvec(arrays::Vector{A}) where A<:CuArray
    T = eltype(arrays[1])
    n = sum(length, arrays)
    dest = CuArray{T}(n)
    offset = 1
    for x in arrays
        copy!(dest, offset, x, 1, length(x))
        offset += length(x)
    end
    dest
end

function test_rnn()
    T = Float32
    x = CuArray(randn(T,1,10,12))
    xs = [randn(T,10,5), randn(T,10,4), randn(T,10,3)]
    xs = map(CuArray, xs)

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
