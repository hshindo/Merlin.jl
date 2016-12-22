function dropout(x, droprate::Float64)
    h = handle(x)
    y = similar(x)
    dropoutdesc = DropoutDesc()
    p = Cint[0]
    cudnnDropoutGetStatesSize(h, p)
    statessize = p[1]
    states = CuArray{Int8}(Int(statessize))

    xdesc = TensorDesc(x)
    ydesc = TensorDesc(y)
    p = Cint[0]
    cudnnDropoutGetReserveSpaceSize(xdesc, p)
    reservesize = p[1]
    reservespace = CuArray{Int8}(Int(reservesize))

    cudnnSetDropoutDescriptor(dropoutdesc, h, Cfloat(droprate), states, statessize, 0)
    cudnnDropoutForward(h, dropoutdesc, xdesc, x, ydesc, y, reservespace, reservesize)

    y, dropoutdesc, reservespace, reservesize
end

function ∇dropout!(dy, dx, droprate, dropoutdesc, reservespace, reservesize)
    h = handle(dy)
    dydesc = TensorDesc(dy)
    dxdesc = TensorDesc(dx)
    cudnnDropoutBackward(h, dropoutdesc, dydesc, dy, dxdesc, dx, reservespace, reservesize)
end
