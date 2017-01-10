function add!(x, y; alpha=1, beta=0)
    p = Ptr{Void}[0]
    T = eltype(x)
    xdesc = TensorDesc(x)
    ydesc = TensorDesc(y)
    cudnnAddTensor(handle(x), T[alpha], xdesc, x, T[beta], ydesc, y)
    y
end
