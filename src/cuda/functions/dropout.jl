function dropout!(out, x::CuArray, rate::Float64)
    dropdesc = CUDNN.DropoutDesc()
    h = CUDNN.HANDLE
    xdesc = CUDNN.TensorDesc(x)
    p = Cint[0]
    CUDNN.cudnnDropoutGetReserveSpaceSize(xdesc, p)
    reservesize = p[1]
    reservespace = CuArray{Int8}(Int(reservesize))

    y = similar(x)
    CUDNN.cudnnDropoutForward(h, dropdesc, xdesc, x, xdesc, y, reservespace, reservesize)

    out.data = y
    out.∇! = function ∇!()
        isvoid(out[1].grad) || CUDNN.cudnnDropoutBackward(h, dropdesc, xdesc, out.grad, xdesc, out[1].grad, reservespace, reservesize)
    end
    out
end

dropout(x::CuArray, rate::Float64) = x
