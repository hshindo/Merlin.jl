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
    work

    function ConvolutionDesc(::Type{T}, pads, strides, dilations) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateConvolutionDescriptor (Ptr{Cptr},) ref
        desc = new(ref[], nothing)
        finalizer(desc, x -> @cudnn :cudnnDestroyConvolutionDescriptor (Cptr,) x.ptr)

        @assert length(pads) == length(strides) == length(dilations)
        N = length(pads)
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
    convdesc = ConvolutionDesc(T, pads, strides, dilations)
    wdesc = FilterDesc(w)
    xdesc = TensorDesc(x)

    ydims = ntuple(N-2) do d
        k = (size(w,d)-1) * dilations[d] + 1
        1 + (size(x,d) + 2pads[d] - k) ÷ strides[d]
    end
    y = similar(x, ydims..., size(w,N), size(x,N))
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

    convdesc.work = (wdesc,xdesc,ydesc)
    y, convdesc
end

function ∇convolution_bias!(dy::CuArray{T}, db::CuArray{T}) where T
    wdesc,xdesc,ydesc = convdesc.work
    @cudnn(:cudnnConvolutionBackwardBias,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, T[1], ydesc, dy, T[1], bdesc, db)
end

function ∇convolution_filter!(convdesc, x::CuArray{T}, dy::CuArray{T}, dw::CuArray{T}) where T
    wdesc,xdesc,ydesc = convdesc.work
    h = gethandle()
    ref = Ref{Cint}()
    preference = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
    @cudnn(:cudnnGetConvolutionBackwardFilterAlgorithm,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Csize_t,Ptr{Cint}),
        h, xdesc, ydesc, convdesc, wdesc, preference, 0, ref)
    algo = ref[]

    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetConvolutionBackwardFilterWorkspaceSize,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Csize_t}),
        h, xdesc, ydesc, convdesc, wdesc, algo, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    @cudnn(:cudnnConvolutionBackwardFilter,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Cptr,Csize_t,Cptr,Cptr,Cptr),
        h, T[1], xdesc, x, ydesc, dy, convdesc, algo, workspace, length(workspace), T[1], wdesc, dw)
end

function ∇convolution_data!(convdesc, w::CuArray{T}, dy::CuArray{T}, dx::CuArray{T}) where T
    wdesc,xdesc,ydesc = convdesc.work
    h = gethandle()
    ref = Ref{Cint}()
    preference = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST
    @cudnn(:cudnnGetConvolutionBackwardDataAlgorithm,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Csize_t,Ptr{Cint}),
        h, wdesc, ydesc, convdesc, xdesc, preference, 0, ref)
    algo = ref[]

    ref = Ref{Csize_t}()
    @cudnn(:cudnnGetConvolutionBackwardDataWorkspaceSize,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Csize_t}),
        h, wdesc, ydesc, convdesc, xdesc, algo, ref)
    workspace = CuArray{UInt8}(Int(ref[]))

    @cudnn(:cudnnConvolutionBackwardData,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Cptr,Csize_t,Cptr,Cptr,Cptr),
        h, T[1], wdesc, w, ydesc, dy, convdesc, algo, workspace, length(workspace), T[1], xdesc, dx)
end
