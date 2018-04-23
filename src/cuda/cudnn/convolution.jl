# cudnnConvolutionMode_t
const CUDNN_CONVOLUTION = Cint(0)
const CUDNN_CROSS_CORRELATION = Cint(1)

# cudnnConvolutionFwdPreference_t
const CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = Cint(0)
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = Cint(1)
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = Cint(2)

# cudnnConvolutionFwdAlgo_t
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = Cint(0)
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = Cint(1)
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = Cint(2)
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = Cint(3)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = Cint(4)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = Cint(5)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = Cint(6)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = Cint(7)
const CUDNN_CONVOLUTION_FWD_ALGO_COUNT = Cint(8)

# cudnnConvolutionBwdFilterPreference_t
const CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = Cint(0)
const CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = Cint(1)
const CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = Cint(2)

# cudnnConvolutionBwdFilterAlgo_t
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = Cint(0)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = Cint(1)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = Cint(2)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = Cint(3)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = Cint(4)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = Cint(5)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = Cint(6)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = Cint(7)

# cudnnConvolutionBwdDataPreference_t
const CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = Cint(0)
const CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = Cint(1)
const CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = Cint(2)

# cudnnConvolutionBwdDataAlgo_t
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = Cint(0)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = Cint(1)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = Cint(2)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = Cint(3)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = Cint(4)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = Cint(5)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = Cint(6)

mutable struct ConvolutionDesc
    ptr::Cptr

    function ConvolutionDesc(::Type{T}, N::Int, pads, strides, dilations) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateConvolutionDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc, x -> @cudnn :cudnnDestroyConvolutionDescriptor (Cptr,) x.ptr)

        cpads = Cint[pads[i] for i=N:-1:1]
        cstrides = Cint[strides[i] for i=N:-1:1]
        cdilations = Cint[dilations[i] for i=N:-1:1]
        mode = CUDNN_CROSS_CORRELATION
        @cudnn(:cudnnSetConvolutionNdDescriptor,
            (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint),
            desc, N, cpads, cstrides, cdilations, mode, datatype(T))
        desc
    end
end

Base.unsafe_convert(::Type{Cptr}, desc::ConvolutionDesc) = desc.ptr

function convolution(w::CuArray{T,N}, x::CuArray{T,N}, pads, strides, dilations) where {T,N}
    @assert size(w,N-1) == size(x,N-1)
    convdesc = ConvolutionDesc(T, N-2, pads, strides, dilations)
    wdesc = FilterDesc(w)
    xdesc = TensorDesc(x)

    ydims = Array{Int}(N)
    for d = 1:N-2
        ydims[d] = 1 + (size(x,d) + 2pads[d] - (((size(w,d)-1)*dilations[d])+1)) ÷ strides[d]
    end
    ydims[N-1] = size(w, N)
    ydims[N] = size(x, N)
    y = similar(x, ydims...)
    ydesc = TensorDesc(y)

    h = gethandle()
    ref = Ref{Cint}()
    preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
    @cudnn(:cudnnGetConvolutionForwardAlgorithm,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Csize_t,Ptr{Cint}),
        h, xdesc, wdesc, convdesc, ydesc, preference, 0, ref)
    algo = ref[]

    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetConvolutionForwardWorkspaceSize,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Csize_t}),
        h, xdesc, wdesc, convdesc, ydesc, algo, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    @cudnn(:cudnnConvolutionForward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Cptr,Csize_t,Cptr,Cptr,Cptr),
        h, T[1], xdesc, x, wdesc, w, convdesc, algo, workspace, length(workspace), T[0], ydesc, y)
    y
end

function ∇convolution!(dy::CuArray{T,N}, w, dw, x, dx, pads, strides, dilations) where {T,N}
    convdesc = ConvolutionDesc(T, N-2, pads, strides, dilations)
    dydesc = TensorDesc(dy)
    wdesc = FilterDesc(w)
    xdesc = TensorDesc(x)
    dw == nothing || backward_filter!(convdesc, xdesc, x, dydesc, dy, wdesc, dw)
    dx == nothing || backward_data!(convdesc, wdesc, w, dydesc, dy, xdesc, dx)
end

function backward_bias!(dydesc, dy, dbdesc, db)
    T = eltype(dy)
    @cudnn(:cudnnConvolutionBackwardBias,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, T[1], dydesc, dy, T[1], dbdesc, db)
end

function backward_filter!(convdesc::ConvolutionDesc, xdesc, x, dydesc, dy, dwdesc, dw)
    T = eltype(x)
    h = gethandle()
    ref = Ref{Cint}()
    preference = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
    @cudnn(:cudnnGetConvolutionBackwardFilterAlgorithm,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Csize_t,Ptr{Cint}),
        h, xdesc, dydesc, convdesc, dwdesc, preference, 0, ref)
    algo = ref[]

    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetConvolutionBackwardFilterWorkspaceSize,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Csize_t}),
        h, xdesc, dydesc, convdesc, dwdesc, algo, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    @cudnn(:cudnnConvolutionBackwardFilter,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Cptr,Csize_t,Cptr,Cptr,Cptr),
        h, T[1], xdesc, x, dydesc, dy, convdesc, algo, workspace, length(workspace), T[1], dwdesc, dw)
end

function backward_data!(convdesc::ConvolutionDesc, wdesc, w, dydesc, dy, dxdesc, dx)
    T = eltype(w)
    h = gethandle()
    ref = Ref{Cint}()
    preference = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST
    @cudnn(:cudnnGetConvolutionBackwardDataAlgorithm,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Csize_t,Ptr{Cint}),
        h, wdesc, dydesc, convdesc, dxdesc, preference, 0, ref)
    algo = ref[]

    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetConvolutionBackwardDataWorkspaceSize,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Csize_t}),
        h, wdesc, dydesc, convdesc, dxdesc, algo, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    @cudnn(:cudnnConvolutionBackwardData,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Cptr,Csize_t,Cptr,Cptr,Cptr),
        h, T[1], wdesc, w, dydesc, dy, convdesc, algo, workspace, length(workspace), T[1], dxdesc, dx)
end
