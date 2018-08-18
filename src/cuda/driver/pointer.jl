export CuPtr

mutable struct CuPtr{T}
    ptr::Ptr{T}
    size::Int
    dev::Int
end

Base.convert(::Type{Ptr{T}}, x::CuPtr) where T = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{Ptr{T}}, x::CuPtr) where T = Ptr{T}(x.ptr)
Base.pointer(x::CuPtr{T}, index::Int=1) where T = x.ptr + sizeof(T)*(index-1)

function memalloc(::Type{T}, size::Int) where T
    @assert size >= 0
    size == 0 && return CuPtr{T}(Ptr{T}(C_NULL), 0, getdevice())
    ref = Ref{Ptr{Cvoid}}()
    @apicall :cuMemAlloc (Ptr{Ptr{Cvoid}},Csize_t) ref sizeof(T)*size
    CuPtr(Ptr{T}(ref[]), size, getdevice())
end

function memfree(ptr::CuPtr)
    ptr.size == 0 && return
    @apicall :cuMemFree (Ptr{Cvoid},) ptr
end

function meminfo()
    ref_free = Ref{Csize_t}()
    ref_total = Ref{Csize_t}()
    @apicall :cuMemGetInfo (Ptr{Csize_t},Ptr{Csize_t}) ref_free ref_total
    Int(ref_free[]), Int(ref_total[])
end
