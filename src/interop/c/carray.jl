immutable CArray{T}
    ptr::Ptr{T}
    N::Cint
    dims::Ptr{Cint}
    strides::Ptr{Cint}
end

CArray(x::Array) = CArray(pointer(x), Cint(ndims(x)), pointer(Cint[size(x)...]), pointer(Cint[strides(x)...]))

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

immutable Cint5
    i1::Cint
    i2::Cint
    i3::Cint
    i4::Cint
    i5::Cint
end

cint(t::NTuple{1,Int}) = Cint1(t[1])
cint(t::NTuple{2,Int}) = Cint2(t[1],t[2])
cint(t::NTuple{3,Int}) = Cint3(t[1],t[2],t[3])
cint(t::NTuple{4,Int}) = Cint4(t[1],t[2],t[3],t[4])
cint(t::NTuple{5,Int}) = Cint5(t[1],t[2],t[3],t[4],t[5])
cint(i::Int...) = cint(i)
