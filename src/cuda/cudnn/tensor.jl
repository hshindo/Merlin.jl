reshape4d{T}(x::CuArray{T,1}) = reshape(x, size(x,1), 1, 1, 1)
reshape4d{T}(x::CuArray{T,2}) = reshape(x, size(x,1), size(x,2), 1, 1)
reshape4d{T}(x::CuArray{T,3}) = reshape(x, size(x,1), size(x,2), size(x,3), 1)
reshape4d{T}(x::CuArray{T,4}) = x

reshape4d_r{T}(x::CuArray{T,1}) = reshape(x, 1, 1, 1, size(x,1))
reshape4d_r{T}(x::CuArray{T,2}) = reshape(x, 1, 1, size(x,1), size(x,2))
reshape4d_r{T}(x::CuArray{T,3}) = reshape(x, 1, size(x,1), size(x,2), size(x,3))
reshape4d_r{T}(x::CuArray{T,4}) = x

function tensor_desc{T,N}(a::CuArray{T,N})
    csize = Cint[size(a,i) for i=ndims(a):-1:1]
    cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
    p = Ptr{Void}[0]
    cudnnCreateTensorDescriptor(p)
    cudnnSetTensorNdDescriptor(p[1], datatype(T), ndims(a), csize, cstrides)
    p[1]
end
