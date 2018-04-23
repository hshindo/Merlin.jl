struct DeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Cint}
    strides::NTuple{N,Cint}
    contigious::Cuchar
end

function DeviceArray(x::CuArray{T}) where T
    DeviceArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)), Cuchar(1))
end

function DeviceArray(x::CuSubArray{T}) where T
    DeviceArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)), Cuchar(0))
end
