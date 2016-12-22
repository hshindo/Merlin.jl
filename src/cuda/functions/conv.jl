import ..Merlin: Conv

function (f::Conv){T<:CuArray}(x::Var{T})
    desc = ConvDesc(T, f.padding, f.strides)
    y = CUDNN.convolution(x.data, w.data, desc)
    function df(gy::CuArray)
        CUDNN.∇convolution_filter!(x.data, gy, desc, w.grad, beta=1.0)
        CUDNN.∇convolution_data!(w.data, gy, desc, x.grad, beta=1.0)
        #CUDNN.∇convolution_bias!(gy, b.grad, beta=0.0)
    end
    Var(y, df, (f.w,x))
end
