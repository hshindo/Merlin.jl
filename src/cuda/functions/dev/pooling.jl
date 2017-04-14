function pooling{T<:CuArray}(x::Var{T}, mode, dims, padding, strides)
    if mode == :max
        mode = CUDNN_POOLING_MAX
    elseif mode == :average
        mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
    end
    y = CUDNN.pooling(T, mode, dims, padding, strides, x)
    df(gy) = CUDNN.∇pooling!(mode, winsize, padding, strides, y, gy, x.data, x.grad)
    Var(y, df, (x,))
end
