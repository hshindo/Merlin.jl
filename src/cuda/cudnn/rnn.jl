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

# cudnnRNNClipMode_t
const CUDNN_RNN_CLIP_NONE = 0
const CUDNN_RNN_CLIP_MINMAX = 1

# cudnnRNNPaddingMode_t
const CUDNN_RNN_PADDED_IO_DISABLED = 0
const CUDNN_RNN_PADDED_IO_ENABLED = 1

mutable struct RNNDesc
    ptr::Cptr
    direction
    work

    function RNNDesc(::Type{T}, hsize::Int, nlayers::Int, droprate::Float64, direction::Cint, mode::Cint) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateRNNDescriptor (Ptr{Cptr},) ref
        desc = new(ref[], direction, nothing)

        h = gethandle()
        dropdesc = DropoutDesc(droprate)
        algo = CUDNN_RNN_ALGO_STANDARD
        @cudnn(:cudnnSetRNNDescriptor,
            (Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),
            h, desc, hsize, nlayers, dropdesc, CUDNN_LINEAR_INPUT, direction, mode, algo, datatype(T))

        finalizer(desc) do x
            @cudnn :cudnnDestroyRNNDescriptor (Cptr,) x.ptr
        end
        desc
    end
end

Base.cconvert(::Type{Cptr}, desc::RNNDesc) = desc.ptr

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
    reserve_space
    x
    batchdims
    y
end

function rnn(insize::Int, hsize::Int, nlayers::Int, droprate::Float64, direction::Cint, mode::Cint,
    w::CuVector{T}, x::CuMatrix{T}, batchdims::Vector{Int}, train::Bool) where T

    @assert insize == size(x,1)
    rnndesc = RNNDesc(T, hsize, nlayers, droprate, direction, mode)
    seqlength = length(batchdims)
    wdesc = FilterDesc(T, 1, 1, length(w))

    # x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength
    # xdesc: Array of T (1,X,B) descriptors
    xdesc = map(batchdims) do d
        TensorDesc(T, 1, insize, d)
    end

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
    # ydesc: Array of T (1,Y,B) descriptors
    y = CuArray{T}(hsize*coef, sum(batchdims))
    ydesc = map(batchdims) do d
        TensorDesc(T, 1, hsize*coef, d)
    end

    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNWorkspaceSize,
        (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
        h, rnndesc, seqlength, xdesc, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    if train
        ref = Ref{Csize_t}()
        @cudnn(:cudnnGetRNNTrainingReserveSize,
            (Cptr,Cptr,Cint,Ptr{Cptr},Ptr{Csize_t}),
            h, rnndesc, seqlength, xdesc, ref)
        reserve_space = CuArray{UInt8}(Int(ref[]))

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
            Cptr,Csize_t),      # reserve_space
            h, rnndesc, seqlength,
            xdesc, x,
            hxdesc, C_NULL,
            cxdesc, C_NULL,
            wdesc, w,
            ydesc, y,
            hydesc, C_NULL,
            cydesc, C_NULL,
            workspace, length(workspace),
            reserve_space, length(reserve_space))
        rnndesc.work = w,x,y,seqlength,wdesc,xdesc,hxdesc,cxdesc,ydesc,workspace,reserve_space
        y, rnndesc
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
            xdesc, x,
            hxdesc, C_NULL,
            cxdesc, C_NULL,
            wdesc, w,
            ydesc, y,
            hydesc, C_NULL,
            cydesc, C_NULL,
            workspace, length(workspace))
        y, rnndesc
    end
end

function ∇rnn_data(rnndesc::RNNDesc, dy::CuArray)
    coef = rnndesc.direction == CUDNN_UNIDIRECTIONAL ? 1 : 2
    w,x,y,seqlength,wdesc,xdesc,hxdesc,cxdesc,ydesc,workspace,reserve_space = rnndesc.work

    dx = similar(x)
    dhy = dcy = hx = cx = dhx = dcx = C_NULL
    hydesc = cydesc = C_NULL
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
        Cptr,Csize_t),  # reserve_space
        gethandle(), rnndesc, seqlength,
        ydesc, y,
        ydesc, dy,
        hydesc, dhy,
        cydesc, dcy,
        wdesc, w,
        hxdesc, hx,
        cxdesc, cx,
        xdesc, dx,
        hxdesc, dhx,
        cxdesc, dcx,
        workspace, length(workspace),
        reserve_space, length(reserve_space))
    dx
end

function ∇rnn_weights!(rnndesc::RNNDesc, dw::CuArray)
    w,x,y,seqlength,wdesc,xdesc,hxdesc,cxdesc,ydesc,workspace,reserve_space = rnndesc.work

    hx = C_NULL
    @cudnn(:cudnnRNNBackwardWeights,
        (Cptr,Cptr,Cint,
        Ptr{Cptr},Cptr,     # x
        Cptr,Cptr,          # hx
        Ptr{Cptr},Cptr,     # y
        Cptr,Csize_t,       # workspace
        Cptr,Cptr,          # dw
        Cptr,Csize_t),      # reserve_space
        gethandle(), rnndesc, seqlength,
        xdesc, x,
        hxdesc, hx,
        ydesc, y,
        workspace, length(workspace),
        wdesc, dw,
        reserve_space, length(reserve_space))
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

function getRNNParamSize(::Type{T}, desc, xdesc) where T
    h = gethandle()
    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetRNNParamsSize,
        (Cptr,Cptr,Cptr,Ptr{Csize_t},Cint),
        h, desc, xdesc, ref, datatype(T))
    println(Int(ref[]) ÷ sizeof(T))
end
