# cudnnTensorFormat_t
const CUDNN_TENSOR_NCHW = Cint(0)
const CUDNN_TENSOR_NHWC = Cint(1)
const CUDNN_TENSOR_NCHW_VECT_C = Cint(2)

mutable struct TensorDesc
    ptr::Cptr

    function TensorDesc(::Type{T}, dims::Union{Vector{Int},Tuple}) where T
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateTensorDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc, x -> @cudnn :cudnnDestroyTensorDescriptor (Cptr,) x.ptr)

        N = length(size)
        strides = Array{Int}(length(dims))
        strides[1] = 1
        for i = 1:length(strides)-1
            strides[i+1] = strides[i] * dims[i]
        end

        csize = Cint[size[i] for i=N:-1:1]
        cstrides = Cint[strides[i] for i=N:-1:1]
        @cudnn(:cudnnSetTensorNdDescriptor,
            (Cptr,Cint,Cint,Ptr{Cint},Ptr{Cint}),
            desc, datatype(T), N, csize, cstrides)
        desc
    end
end

function TensorDesc(x::CuArray{T}, N::Int) where T
    @assert 1 < ndims(x) <= N
    dims = [size(x)...]
    while length(dims) < N
        unshift!(dims, 1)
    end
    TensorDesc(T, dims)
end
TensorDesc(::Type{T}, dims::Int...) where T = TensorDesc(T, dims)

Base.unsafe_convert(::Type{Cptr}, desc::TensorDesc) = desc.ptr
