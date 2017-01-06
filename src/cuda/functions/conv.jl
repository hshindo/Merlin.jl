import ..Merlin: Conv

function (f::Conv){T,N}(x::Var{CuArray{T,N}})
    w, b, pads, strides = f.w, f.b, f.pads, f.strides
    outdims = ntuple(length(pads)) do i
        (size(x.data,i) + 2pads[i] - size(w.data,i)) ÷ strides[i] + 1
    end
    y = similar(x.data, outdims..., size(w.data,N), size(x.data,N))
    desc = ConvDesc(T, pads, strides)
    CUDNN.convolution!(x.data, w.data, desc, y)

    function df(gy::CuArray)
        isa(w.grad, Void) || CUDNN.∇convolution_filter!(x.data, gy, desc, w.grad, beta=1.0)
        isa(x.grad, Void) || CUDNN.∇convolution_data!(w.data, gy, desc, x.grad, beta=1.0)
        isa(b.grad, Void) || CUDNN.∇convolution_bias!(gy, b.grad, beta=1.0)
    end
    Var(y, df, (f.w,x))
end
