export CuPtr

mutable struct CuPtr{T}
    ptr::Ptr{T}
    dev::Int
    size::Int
end

Base.convert(::Type{Ptr{T}}, x::CuPtr) where T = Ptr{T}(x.ptr)
Base.cconvert(::Type{Ptr{T}}, x::CuPtr) where T = Ptr{T}(x.ptr)
# Base.pointer(x::CuPtr{T}, index::Int=1) where T = x.ptr + sizeof(T)*(index-1)
