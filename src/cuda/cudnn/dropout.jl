type DropoutDesc
    ptr::Ptr{Void}
end

function DropoutDesc()
    p = Ptr{Void}[0]
    cudnnCreateDropoutDescriptor(p)
    desc = DropoutDesc(p[1])
    finalizer(desc, cudnnDestroyDropoutDescriptor)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr

function dropout(x, droprate::Float64)
    desc = DropoutDesc()
    h = handle(x)
    p = Cint[0]
    cudnnDropoutGetStatesSize(h, p)
    statessize = p[1]
    states = CuArray{Int8}(Int(statessize))

    p = Cint[0]
    cudnnDropoutGetReserveSpaceSize(xdesc, p)
    reservesize = p[1]
    reservespace = CuArray{Int8}(Int(reservesize))

    xdesc = TensorDesc(x)
    cudnnSetDropoutDescriptor(desc, h, droprate, states, statessize, 0)

    y = similar(x)
    cudnnDropoutForward(h, desc, xdesc, x, xdesc, y, reservespace, reservesize)

    function backward!(gy, gx)
        isvoid(gx) && return
        cudnnDropoutBackward(h, desc, xdesc, dy, xdesc, dx, reservespace, reservesize)
    end
    y, backward!
end
