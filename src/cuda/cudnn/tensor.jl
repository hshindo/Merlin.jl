# cudnnTensorFormat_t
const CUDNN_TENSOR_NCHW = 0
const CUDNN_TENSOR_NHWC = 1
const CUDNN_TENSOR_NCHW_VECT_C = 2

mutable struct TensorDesc
    ptr::Cptr

    function TensorDesc(::Type{T}, dims::Dims{N}) where {T,N}
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateTensorDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc) do x
            @cudnn :cudnnDestroyTensorDescriptor (Cptr,) x.ptr
        end

        strides = Array{Int}(undef, N)
        strides[1] = 1
        for i = 1:length(strides)-1
            strides[i+1] = strides[i] * dims[i]
        end

        csize = Cint[dims[i] for i=N:-1:1]
        cstrides = Cint[strides[i] for i=N:-1:1]
        @cudnn(:cudnnSetTensorNdDescriptor,
            (Cptr,Cint,Cint,Ptr{Cint},Ptr{Cint}),
            desc, datatype(T), N, csize, cstrides)
        desc
    end
end
TensorDesc(::Type{T}, dims::Int...) where T = TensorDesc(T, dims)

function TensorDesc(x::CuArray{T}, N::Int=ndims(x)) where T
    @assert ndims(x) <= N && N > 1
    dims = [size(x)...]
    ndims(x) == 1 && push!(dims,1)
    while length(dims) < N
        pushfirst!(dims, 1)
    end
    TensorDesc(T, tuple(dims...))
end

Base.cconvert(::Type{Cptr}, desc::TensorDesc) = desc.ptr
