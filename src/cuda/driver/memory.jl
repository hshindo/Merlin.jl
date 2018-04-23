export MemBlock

mutable struct MemBlock
    ptr::Ptr{Void}
    bytesize::Int
    dev::Int
end

MemBlock(ptr::Ptr{Void}, bytesize::Int) = MemBlock(ptr, bytesize, getdevice())

Base.convert(::Type{Ptr{T}}, x::MemBlock) where T = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{Ptr{T}}, x::MemBlock) where T = Ptr{T}(x.ptr)

function memalloc(bytesize::Int)
    ref = Ref{Ptr{Void}}()
    @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref bytesize
    ref[]
end

function memfree(ptr::Ptr{Void})
    @apicall :cuMemFree (Ptr{Void},) ptr
end

function meminfo()
    ref_free = Ref{Csize_t}()
    ref_total = Ref{Csize_t}()
    @apicall :cuMemGetInfo (Ptr{Csize_t},Ptr{Csize_t}) ref_free ref_total
    Int(ref_free[]), Int(ref_total[])
end
