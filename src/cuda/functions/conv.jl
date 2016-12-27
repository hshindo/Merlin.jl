import ..Merlin: Conv

function (f::Conv){T,N}(x::Var{CuArray{T,N}})
    w, padding, strides = f.w, f.padding, f.strides
    outdims = ntuple(length(padding)) do i
        (size(x,i) + 2padding[i] - size(w,i)) ÷ strides[i] + 1
    end
    y = similar(x, outdims..., size(w.data,N+2), size(x.data,N+2))
    desc = ConvDesc(T, padding, strides)
    CUDNN.convolution!(x.data, w.data, desc, y)

    function df(gy::CuArray)
        isvoid(w.grad) || CUDNN.∇convolution_filter!(x.data, gy, desc, w.grad, beta=1.0)
        isvoid(x.grad) || CUDNN.∇convolution_data!(w.data, gy, desc, x.grad, beta=1.0)
        isvoid(b.grad) || CUDNN.∇convolution_bias!(gy, b.grad, beta=0.0)
    end
    Var(y, df, (f.w,x))
end
