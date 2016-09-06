function tensor_desc{T,N}(a::CuArray{T,N})
    if N < 4
        # might be inefficient
        s = [1,1,1,1]
        for i = 1:N
            s[i] = size(a, i)
        end
        a = reshape(a, tuple(s...))
    end
    csize = Cint[size(a,i) for i=ndims(a):-1:1]
    cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
    p = Ptr{Void}[0]
    cudnnCreateTensorDescriptor(p)
    cudnnSetTensorNdDescriptor(p[1], datatype(T), ndims(a), csize, cstrides)
    p[1]
end
