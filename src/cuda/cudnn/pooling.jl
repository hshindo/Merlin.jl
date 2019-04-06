# cudnnPoolingMode_t
const CUDNN_POOLING_MAX = 0
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
const CUDNN_POOLING_MAX_DETERMINISTIC = 3

mutable struct PoolingDesc
    ptr::Cptr

    function PoolingDesc(mode::Int, window::NTuple, padding::NTuple, stride::NTuple)
        ref = Ref{Cptr}()
        @cudnn :cudnnCreatePoolingDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc) do x
            @cudnn :cudnnDestroyPoolingDescriptor (Cptr,) x.ptr
        end

        nanopt = CUDNN_PROPAGATE_NAN
        N = length(window)
        cwindow = Cint[window[i] for i=N:-1:1]
        cpadding = Cint[padding[i] for i=N:-1:1]
        cstride = Cint[stride[i] for i=N:-1:1]
        @cudnn(:cudnnSetPoolingNdDescriptor,
            (Cptr,Cint,Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
            desc, mode, nanopt, N, cwindow, cpadding, cstride)
        desc
    end
end

Base.cconvert(::Type{Cptr}, desc::PoolingDesc) = desc.ptr

function pooling(x::CuArray{T,N}, mode, window, padding, stride) where {T,N}
    @assert length(window) == N-2
    @assert length(padding) == N-2
    @assert length(stride) == N-2

    pdesc = PoolingDesc(mode, window, padding, stride)
    ydims = ntuple(N-2) do d
        1 + (size(x,d) + 2padding[d] - window[d]) ÷ stride[d]
    end

    xdesc = TensorDesc(x, N)
    y = similar(x, ydims..., size(x,N-1), size(x,N))
    ydesc = TensorDesc(y, N)
    h = gethandle()
    @cudnn(:cudnnPoolingForward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, pdesc, T[1], xdesc, x, T[0], ydesc, y)
    y, pdesc
end

function ∇pooling!(y::CuArray{T,N}, dy, x, dx, pdesc::PoolingDesc) where {T,N}
    xdesc = TensorDesc(x, N)
    ydesc = TensorDesc(y, N)
    h = gethandle()
    @cudnn(:cudnnPoolingBackward,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, pdesc, T[1], ydesc, y, ydesc, dy, xdesc, x, T[1], xdesc, dx)
end
