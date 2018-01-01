function relu!(out, x::CuArray)
    work = CUDNN.relu(x)
    out.data = work.y
    out.∇! = () -> begin
        isvoid(out[1].grad) && return
        CUDNN.∇activation!(work, out.grad, out[1].grad)
    end
    out
end
