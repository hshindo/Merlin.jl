export avgpooling1d

function avgpooling1d(x::Var, dims::Vector{Int}; ksize::Int, padding=0, stride=1)
    x = pack(x, dims, 0)
    ydata, work = avgpooling1d(x.data, ksize, padding, stride)
    y = Var(ydata, ∇avgpooling1d!, (x,work))
    unpack(y, dims)
end

function avgpooling1d(x::CuArray{T,3}, ksize, padding, stride) where T
    x = reshape(x, size(x,1), size(x,2), 1, size(x,3))
    ksize = (1, ksize)
    padding = (0, padding)
    stride = (1, stride)
    y, work = CUDNN.pooling(x, CUDNN.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, ksize, padding, stride)
    dropdims(y,dims=3), work
end

function ∇avgpooling1d!(y::Var, x::Var, work)
    isnothing(y.grad) && return
    ydata = reshape(y.data, size(y,1), size(y,2), 1, size(y,3))
    ygrad = reshape(y.grad, size(ydata)...)
    xdata = reshape(x.data, size(x,1), size(x,2), 1, size(x,3))
    xgrad = reshape(x.grad, size(xdata)...)
    ∇avgpooling1d!(ydata, ygrad, xdata, xgrad, work)
end

function ∇avgpooling1d!(y::CuArray, gy, x, gx, work)
    CUDNN.∇pooling!(y, gy, x, gx, work)
end
