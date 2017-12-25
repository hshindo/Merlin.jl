module CUDNN

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_7"])
else
    const libcudnn = Libdl.find_library(["libcudnn"])
end
if isempty(libcudnn)
    warn("CUDNN library cannot be found.")
end

const API_VERSION = ccall((:cudnnGetVersion,libcudnn),Cint,())
info("CUDNN API $API_VERSION")

macro apicall(f, args...)
    quote
        status = ccall(($(QuoteNode(f)),libcuda), Cint, $(map(esc,args)...))
        if status != 0
            Base.show_backtrace(STDOUT, backtrace())
            p = ccall((:cudnnGetErrorString,libcudnn), Cstring, (Cint,), status)
            throw(unsafe_string(p))
        end
    end
end

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF
datatype(::Type{Int8}) = CUDNN_DATA_INT8
datatype(::Type{Int32}) = CUDNN_DATA_INT32

function init()
    # Setup default cudnn handle
    ref = Ref{Void}()
    cudnnCreate(ref)
    global HANDLE = ref[]
    # destroy cudnn handle at julia exit
    atexit(() -> cudnnDestroy(HANDLE))
end
init()

mutable struct ActivationDesc
    ptr::Ptr{Void}

    function ActivationDesc(mode::UInt32)
        p = Ptr{Void}[0]
        cudnnCreateActivationDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyActivationDescriptor)
        cudnnSetActivationDescriptor(desc, mode, CUDNN_NOT_PROPAGATE_NAN, 1.0)
        desc
    end
end

mutable struct TensorDesc
    ptr::Ptr{Void}

    function TensorDesc(x::CuArray{T,N}; pad=0) where {T,N}
        csize = Cint[1, 1, 1, 1]
        cstrides = Cint[1, 1, 1, 1]
        st = strides(x)
        for i = 1:N
            csize[4-i-pad+1] = size(x,i)
            cstrides[4-i-pad+1] = st[i]
        end
        p = Ptr{Void}[0]
        cudnnCreateTensorDescriptor(p)
        cudnnSetTensorNdDescriptor(p[1], datatype(T), length(csize), csize, cstrides)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyTensorDescriptor)
        desc
    end
end

mutable struct DropoutDesc
    ptr::Ptr{Void}

    function DropoutDesc()
        p = Ptr{Void}[0]
        cudnnCreateDropoutDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyDropoutDescriptor)

        p = Cint[0]
        cudnnDropoutGetStatesSize(h, p)
        statessize = p[1]
        states = CuArray{Int8}(Int(statessize))
        CUDNN.cudnnSetDropoutDescriptor(desc, HANDLE, rate, states, statessize, 0)
        desc
    end
end

mutable struct ReduceTensorDesc
    ptr::Ptr{Void}

    function ReduceTensorDesc(T::Type, op)
        p = Ptr{Void}[0]
        cudnnCreateReduceTensorDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyReduceTensorDescriptor)

        indices = op == CUDNN_REDUCE_TENSOR_MIN || op == CUDNN_REDUCE_TENSOR_MAX ?
            CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
            CUDNN_REDUCE_TENSOR_NO_INDICES
        cudnnSetReduceTensorDescriptor(desc, op, datatype(T), CUDNN_NOT_PROPAGATE_NAN, indices, CUDNN.CUDNN_32BIT_INDICES)
        desc
    end
end

mutable struct RNNDesc
    ptr::Ptr{Void}

    function RNNDesc()
        p = Ptr{Void}[0]
        cudnnCreateRNNDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyRNNDescriptor)
        desc
    end
end

mutable struct FilterDesc
    ptr::Ptr{Void}

    function FilterDesc(x::CuArray{T}, dims) where T
        csize = Cint[size(x,i) for i=N:-1:1]
        p = Ptr{Void}[0]
        cudnnCreateFilterDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyFilterDescriptor)

        cudnnSetFilterNdDescriptor(desc, datatype(T), CUDNN.CUDNN_TENSOR_NCHW, length(dims), dims)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::ReduceTensorDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::RNNDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::FilterDesc) = desc.ptr

end
