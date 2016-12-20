function convolution{T<:CuArray}(::Type{T}, x::Var, w::Var, padding, strides)
    y = CUDNN.convolution(x.data, w.data, padding, strides)
    function df(gy::CuArray)
        CUDNN.∇convolution_filter!(x.data, padding, strides, gy, w.grad, beta=1.0)
        CUDNN.∇convolution_data!(w.data, padding, strides, gy, x.grad, beta=1.0)
    end
    Var(y, df, (x,w))
end
