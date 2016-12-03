function filter_desc{T}(a::CuArray{T}, format=CUDNN_TENSOR_NCHW)
    csize = Cint[size(a,i) for i=ndims(a):-1:1]
    p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(p)
    cudnnSetFilterNdDescriptor(p[1], datatype(T), format, ndims(a), csize)
    p[1]
end
