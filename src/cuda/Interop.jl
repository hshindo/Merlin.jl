module Interop

import ..CUDA: CuArray, CuSubArray, AbstractCuArray

immutable Array{T,N}
    ptr::Ptr{T}
    dims::N
    strides::N
    continuous::Bool
end

Array{T,N}(x::CuArray{T,N}) = Array(pointer(x),cint(size(x)),cint(strides(x)),true)
Array{T,N}(x::CuSubArray{T,N}) = Array(pointer(x),cint(size(x)),cint(strides(x)),false)

immutable Cint1
    i1::Cint
end
immutable Cint2
    i1::Cint
    i2::Cint
end
immutable Cint3
    i1::Cint
    i2::Cint
    i3::Cint
end
immutable Cint4
    i1::Cint
    i2::Cint
    i3::Cint
    i4::Cint
end

cint(t::NTuple{1,Int}) = Cint1(t[1])
cint(t::NTuple{2,Int}) = Cint2(t[1],t[2])
cint(t::NTuple{3,Int}) = Cint3(t[1],t[2],t[3])
cint(t::NTuple{4,Int}) = Cint4(t[1],t[2],t[3],t[4])

end
