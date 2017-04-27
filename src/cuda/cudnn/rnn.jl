export
    # cudnnRNNInputMode_t
    CUDNN_LINEAR_INPUT,
    CUDNN_SKIP_INPUT,

    # cudnnDirectionMode_t
    CUDNN_UNIDIRECTIONAL,
    CUDNN_BIDIRECTIONAL,

    # cudnnRNNMode_t
    CUDNN_RNN_RELU,
    CUDNN_RNN_TANH,
    CUDNN_LSTM,
    CUDNN_GRU

type RNNDesc
    ptr::Ptr{Void}
    workspace
end

function RNNDesc{T}(::Type{T}, dropoutdesc, dir, mode, h, seqlength::Int)
    p = Ptr{Void}[0]
    cudnnCreateRNNDescriptor(p)
    ptr = p[1]

    cudnnSetRNNDescriptor(ptr, hsize, nlayers, dropoutdesc, CUDNN_LINEAR_INPUT, dir, mode, datatype(T))

    p = Csize_t[0]
    cudnnGetRNNWorkspaceSize(h, ptr, seqlength, xdesc, p)
    workspace = CuArray{Csize_t}(p[1])

    desc = RNNDesc(ptr, workspace)
    finalizer(desc, cudnnDestroyRNNDescriptor)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::RNNDesc) = desc.ptr
