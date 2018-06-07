mutable struct CuPtr{T}
    ptr::Ptr{T}
    size::Int
    dev::Int
end

Base.convert(::Type{Ptr{T}}, x::CuPtr) where T = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{Ptr{T}}, x::CuPtr) where T = Ptr{T}(x.ptr)
Base.pointer(x::CuPtr{T}, index::Int=1) where T = x.ptr + sizeof(T)*(index-1)

function memalloc(bytesize::Int)
    ref = Ref{Ptr{Void}}()
    @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref bytesize
    ref[]
end
function memalloc(::Type{T}, size::Int) where T
    ptr = Ptr{T}(memalloc(sizeof(T)*size))
    CuPtr(ptr, size, getdevice())
end

function memfree(ptr::CuPtr)
    @apicall :cuMemFree (Ptr{Void},) ptr
end

function meminfo()
    ref_free = Ref{Csize_t}()
    ref_total = Ref{Csize_t}()
    @apicall :cuMemGetInfo (Ptr{Csize_t},Ptr{Csize_t}) ref_free ref_total
    Int(ref_free[]), Int(ref_total[])
end
