export
    # cudnnConvolutionMode_t
    CUDNN_CONVOLUTION,
    CUDNN_CROSS_CORRELATION,

    # cudnnConvolutionFwdPreference_t
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,

    cudnnGetConvolutionForwardAlgorithm,
    cudnnGetConvolutionForwardWorkspaceSize,
    cudnnConvolutionForward,

    cudnnConvolutionBackwardBias,
    cudnnGetConvolutionBackwardFilterAlgorithm,
    cudnnGetConvolutionBackwardFilterWorkspaceSize,
    cudnnConvolutionBackwardFilter,
    cudnnGetConvolutionBackwardDataAlgorithm,
    cudnnGetConvolutionBackwardDataWorkspaceSize,
    cudnnConvolutionBackwardData

type ConvDesc
    ptr::Ptr{Void}
end

function ConvDesc{T,N}(::Type{T}, pads::NTuple{N,Int}, strides; mode=CUDNN_CROSS_CORRELATION)
    p = Ptr{Void}[0]
    cudnnCreateConvolutionDescriptor(p)
    c_pads = Cint[pads[i] for i=N:-1:1]
    c_strides = Cint[strides[i] for i=N:-1:1]
    c_upscale = fill(Cint(1), N)
    cudnnSetConvolutionNdDescriptor(p[1], N, c_pads, c_strides, c_upscale, mode, datatype(T))
    desc = new(p[1])
    finalizer(desc, cudnnDestroyConvolutionDescriptor)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvDesc) = desc.ptr
