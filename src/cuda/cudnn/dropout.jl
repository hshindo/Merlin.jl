function dropout(x, droprate)
    h = handle(x)
    y = similar(x)
    p = Ptr{Void}[0]
    cudnnCreateDropoutDescriptor(p)
    dropoutdesc = p[1]

    statessize_p = Cint[0]
    cudnnDropoutGetStatesSize(h, statessize_p)
    statessize = statessize_p[1]
    states = CuArray(Int8, Int(statessize))

    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    reservesize_p = Cint[0]
    cudnnDropoutGetReserveSpaceSize(xdesc, reservesize_p)
    reservesize = reservesize_p[1]
    reservespace = CuArray(Int8, Int(reservesize))

    states_p = Ptr{Void}[0]
    cudnnSetDropoutDescriptor(dropoutdesc, h, Cfloat(droprate), states, statessize, 0)
    cudnnDropoutForward(h, dropoutdesc, xdesc, x, ydesc, y, reservespace, reservesize)

    cudnnDestroyDropoutDescriptor(dropoutdesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    y, states, statessize, reservespace, reservesize
end

function ∇dropout!(dy, dx, droprate, states, statessize, reservespace, reservesize)
    h = handle(dy)
    p = Ptr{Void}[0]
    cudnnCreateDropoutDescriptor(p)
    dropoutdesc = p[1]

    dydesc = tensor_desc(dy)
    dxdesc = tensor_desc(dx)
    cudnnSetDropoutDescriptor(dropoutdesc, h, Cfloat(droprate), states, statessize, 0)
    cudnnDropoutBackward(h, dropoutdesc, dydesc, dy, dxdesc, dx, reservespace, reservesize)

    cudnnDestroyDropoutDescriptor(dropoutdesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(dxdesc)
    dx
end
