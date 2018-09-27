export CuDeviceArray

struct CuDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Cint}
    strides::NTuple{N,Cint}
end

function CuDeviceArray(x::CuArray)
    CuDeviceArray(pointer(x).ptr, Cint.(size(x)), Cint.(strides(x)))
end
function CuDeviceArray(x::CuSubArray)
    CuDeviceArray(pointer(x).ptr, Cint.(size(x)), Cint.(strides(x)))
end

Base.size(x::CuDeviceArray) = Int.(x.dims)
Base.length(x::CuDeviceArray) = Int(prod(x.dims))
