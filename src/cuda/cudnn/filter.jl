mutable struct FilterDesc
    ptr::Cptr

    function FilterDesc(::Type{T}, dims::Dims{N}) where {T,N}
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateFilterDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        push!(CUDA.ALLOCATED, desc)
        finalizer(desc) do x
            @cudnn :cudnnDestroyFilterDescriptor (Cptr,) x.ptr
        end

        csize = Cint[reverse(dims)...]
        push!(CUDA.ALLOCATED, csize)
        @cudnn(:cudnnSetFilterNdDescriptor,
            (Cptr,Cint,Cint,Cint,Ptr{Cint}),
            desc, datatype(T), CUDNN_TENSOR_NCHW, N, csize)
        desc
    end
end
FilterDesc(::Type{T}, dims::Int...) where T = FilterDesc(T, dims)
FilterDesc(x::CuArray) = FilterDesc(eltype(x), size(x))

Base.cconvert(::Type{Cptr}, desc::FilterDesc) = desc.ptr
