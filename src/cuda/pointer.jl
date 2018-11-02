export CuPtr

struct CuPtr{T}
    ptr::Ptr{T}
end

Base.cconvert(::Type{Ptr{T}}, x::CuPtr{T}) where T = x.ptr

#=
mutable struct CuPtr{T}
    ptr::Ptr{T}
    size::Int
    dev::Int
end

CuPtr(ptr::Ptr{T}, size::Int) where T = CuPtr(ptr, size, getdevice())
CuPtr{T}() where T = CuPtr(Ptr{T}(C_NULL), 0)

Base.cconvert(::Type{Ptr{T}}, x::CuPtr) where T = Ptr{T}(x.ptr)
Base.pointer(x::CuPtr{T}, index::Int=1) where T = x.ptr + sizeof(T)*(index-1)
=#
