function dropout!(out, x::CuArray, droprate::Float64)
    work = CUDNN.dropout(x, droprate)
    out.data = work.y
    out.∇! = () -> begin
        isvoid(out[1].grad) && return
        ∇dropout!(work, out.grad, out[1].grad)
    end
    out
end
