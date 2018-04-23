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

    function RNNDesc(::Type{T}, hsize::Int, nlayers::Int, droprate::Float64, direction::Cint, mode::Cint) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateRNNDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])

        h = gethandle()
        dropdesc = DropoutDesc(droprate)
        algo = CUDNN_RNN_ALGO_STANDARD
        @cudnn(:cudnnSetRNNDescriptor,
            (Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
            h, desc, hsize, nlayers, dropdesc, CUDNN_LINEAR_INPUT, direction, mode, algo, datatype(T))

        finalizer(desc, x -> @cudnn :cudnnDestroyRNNDescriptor (Cptr,) x.ptr)
        desc
    end
end

Base.unsafe_convert(::Type{Cptr}, desc::RNNDesc) = desc.ptr

struct RNNWork
    direction
    seqlength
    rnndesc
    xdesc
    wdesc
    hxdesc
    cxdesc
    ydesc
    hydesc
    cydesc
    workspace
    reservespace
    x
    batchdims
    y
end

function rnn(insize::Int, hsize::Int, nlayers::Int, droprate::Float64, direction::Cint, mode::Cint,
    w::CuVector{T}, x::CuMatrix{T}, batchdims::Vector{Int}; training=true) where T

    rnndesc = RNNDesc(T, hsize, nlayers, droprate, direction, mode)
    @assert insize == size(x,1)
    seqlength = length(batchdims)
    wdesc = FilterDesc(T, 1, 1, length(w))

    # x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
    # xDesc: Array of T (1,X,B) descriptors
    xdesc = map(batchdims) do d
        TensorDesc(T, 1, insize, d)
    end
    xdesc_ptr = map(d -> d.ptr, xdesc)

    # hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)
    coef = direction == CUDNN_UNIDIRECTIONAL ? 1 : 2
    hxdesc = cxdesc = hydesc = cydesc = C_NULL
    #hxdesc = TensorDesc(T, hsize, batchdims[1], nlayers*coef)
    #cxdesc = TensorDesc(T, hsize, batchdims[1], nlayers*coef)
    #hydesc = TensorDesc(T, hsize, batchdims[1], nlayers*coef)
    #cydesc = TensorDesc(T, hsize, batchdims[1], nlayers*coef)
    # hy = zeros(rnn.hx)
    # cy = zeros(rnn.cx)
    hy = C_NULL
    cy = C_NULL

    # y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)
    # yDesc: Array of T (1,Y,B) descriptors
    y = CuArray{T}(hsize*coef, sum(batchdims))
    ydesc = map(batchdims) do d
        TensorDesc(T, 1, hsize*coef, d)
    end
    ydesc_ptr = map(d -> d.ptr, ydesc)

    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNWorkspaceSize,
        (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
        h, rnndesc, seqlength, xdesc_ptr, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    if training
        ref = Ref{Csize_t}()
        @cudnn(:cudnnGetRNNTrainingReserveSize,
            (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
            h, rnndesc, seqlength, xdesc_ptr, ref)
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
            h, rnndesc, seqlength,
            xdesc_ptr, x,
            hxdesc, C_NULL,
            cxdesc, C_NULL,
            wdesc, w,
            ydesc_ptr, y,
            hydesc, C_NULL,
            cydesc, C_NULL,
            workspace, length(workspace),
            reservespace, length(reservespace))
        work = RNNWork(direction, seqlength, rnndesc, xdesc, wdesc, hxdesc, cxdesc, ydesc, hydesc, cydesc, workspace, reservespace, x, batchdims, y)
        y, work
    else
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
            h, rnndesc, seqlength,
            xdesc_ptr, x,
            hxdesc, C_NULL,
            cxdesc, C_NULL,
            wdesc, w,
            ydesc_ptr, y,
            hydesc, C_NULL,
            cydesc, C_NULL,
            workspace, length(workspace))
        y, nothing
    end
end

function backward_data(y::CuArray, dy::CuArray, x::CuArray{T}, w::CuArray, work::RNNWork) where T
    h = gethandle()
    coef = work.direction == CUDNN_UNIDIRECTIONAL ? 1 : 2
    seqlength = work.seqlength
    rnndesc = work.rnndesc

    xdesc, ydesc = work.xdesc, work.ydesc
    xdesc_ptr = map(d -> d.ptr, xdesc)
    ydesc_ptr = map(d -> d.ptr, ydesc)

    dx = similar(x)
    dxdesc_ptr = xdesc_ptr
    dydesc_ptr = ydesc_ptr

    #dhxdesc = TensorDesc(T, rnn.hsize, work.batchdims[1], rnn.nlayers*coef)
    #dcxdesc = TensorDesc(T, rnn.hsize, work.batchdims[1], rnn.nlayers*coef)
    dhy = dcy = hx = cx = dhx = dcx = C_NULL
    dhxdesc = dcxdesc = C_NULL
    @cudnn(:cudnnRNNBackwardData,
        (Cptr,Cptr,Cint,
        Ptr{Cptr},Cptr,     # y
        Ptr{Cptr},Cptr,     # dy
        Cptr,Cptr,  # dhy
        Cptr,Cptr,  # dcy
        Cptr,Cptr,  # w
        Cptr,Cptr,  # hx
        Cptr,Cptr,  # cx
        Ptr{Cptr},Cptr,  # dx
        Cptr,Cptr,  # dhx
        Cptr,Cptr,  # dcx
        Cptr,Csize_t,   # workspace
        Cptr,Csize_t),  # reservespace
        h, rnndesc, seqlength,
        ydesc_ptr, y,
        ydesc_ptr, dy,
        work.hydesc, dhy,
        work.cydesc, dcy,
        work.wdesc, w,
        work.hxdesc, hx,
        work.cxdesc, cx,
        dxdesc_ptr, dx,
        dhxdesc, dhx,
        dcxdesc, dcx,
        work.workspace, length(work.workspace),
        work.reservespace, length(work.reservespace))
    dx
end

function backward_weights!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, dw::CuArray{T}, work::RNNWork) where T
    h = gethandle()
    hx = C_NULL
    seqlength = work.seqlength
    rnndesc = work.rnndesc
    xdesc, ydesc = work.xdesc, work.ydesc
    xdesc_ptr = map(d -> d.ptr, xdesc)
    ydesc_ptr = map(d -> d.ptr, ydesc)
    dwdesc = work.wdesc
    @cudnn(:cudnnRNNBackwardWeights,
        (Cptr,Cptr,Cint,
        Ptr{Cptr},Cptr,     # x
        Cptr,Cptr,          # hx
        Ptr{Cptr},Cptr,     # y
        Cptr,Csize_t,       # workspace
        Cptr,Cptr,          # dw
        Cptr,Csize_t),      # reservespace
        h, rnndesc, seqlength,
        xdesc_ptr, x,
        work.hxdesc, hx,
        ydesc_ptr, y,
        work.workspace, length(work.workspace),
        dwdesc, dw,
        work.reservespace, length(work.reservespace))
end

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

function split(x::CuArray{T,N}, dim::Int, splitsize::Vector{Int}) where {T,N}
    dims = Any[Colon() for i=1:N]
    offset = 0
    map(splitsize) do s
        dims[dim] = offset+1:offset+s
        offset += s
        view(x, dims...)
    end
end

function getRNNParamSize(::Type{T}, desc, xdesc) where T
    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNParamsSize,
        (Cptr,Cptr,Cptr,Ptr{Csize_t},Cint),
        h, desc, xdesc, ref, datatype(T))
    println(Int(ref[]) ÷ sizeof(T))
end
