function relu!(out, x::CuArray)
    #y = malloc(out, T, size(x))
    #out.work = CUDNN.relu!(x, y)
    actdesc, xdesc, y = CUDNN.relu(x)
    out.data = y
    out.work = actdesc, xdesc
    out
end

function ∇relu!(y::CuArray, gy, x, gx, actdesc, xdesc)
    CUDNN.∇activation!(actdesc, xdesc, y, gy, x, gx)
end

function sigmoid!(out, x::CuArray)
    actdesc, xdesc, y = CUDNN.sigmoid(x)
    out.data = y
    out.work = actdesc, xdesc
    out
end

function ∇sigmoid!(y::CuArray, gy, x, gx, actdesc, xdesc)
    CUDNN.∇activation!(actdesc, xdesc, y, gy, x, gx)
end

function tanh!(out, x::CuArray)
    actdesc, xdesc, y = CUDNN.tanh(x)
    out.data = y
    out.work = actdesc, xdesc
    out
end

function ∇tanh!(y::CuArray, gy, x, gx, actdesc, xdesc)
    CUDNN.∇activation!(actdesc, xdesc, y, gy, x, gx)
end
