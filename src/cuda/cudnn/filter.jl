mutable struct FilterDesc
    ptr::Cptr

    function FilterDesc(::Type{T}, dims::NTuple{N,Int}) where {T,N}
        ref = Ref{Cptr}()
        @cudnn :cudnnCreateFilterDescriptor (Ptr{Cptr},) ref
        desc = new(ref[])
        finalizer(desc, x -> @cudnn :cudnnDestroyFilterDescriptor (Cptr,) x.ptr)

        csize = Cint[reverse(dims)...]
        @cudnn(:cudnnSetFilterNdDescriptor,
            (Cptr,Cint,Cint,Cint,Ptr{Cint}),
            desc, datatype(T), CUDNN_TENSOR_NCHW, N, csize)
        desc
    end
end

FilterDesc(::Type{T}, dims::Int...) where T = FilterDesc(T, dims)
FilterDesc(x::CuArray) = FilterDesc(eltype(x), size(x))

Base.unsafe_convert(::Type{Cptr}, desc::FilterDesc) = desc.ptr
