import Merlin.Conv
import Base.conv

function (f::Conv){T<:CuArray}(x::Var{T})
    y = CUDNN.convolution(x.data, w.data, f.padding, f.strides)
    function df(gy::CuArray)
        CUDNN.∇convolution_filter!(x.data, padding, strides, gy, w.grad, beta=1.0)
        CUDNN.∇convolution_data!(w.data, padding, strides, gy, x.grad, beta=1.0)
    end
    Var(y, df, (f.w,x))
end
