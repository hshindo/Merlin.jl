export Conv2d

"""
    Conv2d(T, filtersize, kwargs...)

* W: (W1,W2,...,I,O)
* X: (X1,X2,...,I,N)
* Y: (Y1,Y2,...,O,N)

where
* I: number of input channels
* O: number of output channels
* N: batch size

```julia
T = Float32
conv = Conv2d(T, (1,1,3,2))
x = CuArray{T}(5,5,3,3)
y = conv(x)
```
"""
mutable struct Conv2d
    w::Var
    pad::NTuple{2,Int}
    stride::NTuple{2,Int}
    dilation::NTuple{2,Int}
end

function Conv2d(::Type{T}, filtersize::Tuple;
    pad=0, stride=1, dilation=1, init_w=Xavier(), init_b=Fill(0)) where T

    N = length(filtersize) - 2
    isa(pad,Int) && (pad = ntuple(_ -> pad, N))
    isa(stride,Int) && (stride = ntuple(_ -> stride, N))
    isa(dilation,Int) && (dilation = ntuple(_ -> dilation, N))

    w = init_w(T, filtersize...)
    Conv(param(w), pad, stride, dilation)
end

function (f::Conv2d)(x::Var)
    configure!(f.w, x)
    y, work = f(x.data)
    Var(y, (f,x,work))
end

function (f::Conv2d)(x::Array{T,4}) where T
    hdims = ntuple(2) do d
        k = (size(w,d)-1) * dilation[d] + 1
        1 + (size(x,d) + 2pad[d] - k) ÷ stride[d]
    end
    h = similar(x, hdims)
    #im2col(x, h, )
    y = linear(h, f.W.data, f.b.data)
    Var(y, (f,x,f.W,f.b,batchdims,h))
end

function (f::Conv2d)(x::CuArray)
    CUDNN.convolution(f.w.data, x, f.pad, f.stride, f.dilation)
end

function addgrad!(y::Var, f::Conv2d, x::Var, work)
    isvoid(x.grad) && return
    ∇conv!(y.grad, f, x.data, x.grad, work)
end

function ∇conv!(gy::CuArray, f::Conv2d, x::CuArray, gx, convdesc)
    w, gw = f.w.data, f.w.grad
    isvoid(gw) || CUDNN.∇convolution_filter!(convdesc, x, gy, gw)
    isvoid(gx) || CUDNN.∇convolution_data!(convdesc, w, gy, gx)
end

function im2col{T}(x::Array{T,4}, kernel::NTuple{4,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int}, dilation::NTuple{2,Int})
    width, height, channels, num = size(x)
    kernel_w, kernel_h = kernel[1], kernel[2]
    pad_w, pad_h = pad
    stride_w, stride_h = stride

    y_w = (width + 2pad_w - (dilation_w * (kernel_w-1) + 1)) ÷ stride_w + 1
    y_h = (height + 2pad_h - (dilation_h * (kernel_h-1) + 1)) ÷ stride_h + 1
    y = similar(x, y_w, y_h, kernel[4], num)

    height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
    width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
    channels_col = channels * kernel_h * kernel_w

    for c = 0:channels_col-1
        w_offset = c % kernel_w
        h_offset = div(c, kernel_w) % kernel_h
        c_im = div(c, kernel_h * kernel_w) # channel
        for h = 0:height_col-1
            for w = 0:width_col-1
                h_pad = h*stride_h - pad_h + h_offset
                w_pad = w*stride_w - pad_w + w_offset
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    @inbounds col[1 + (c*height_col+h) * width_col + w] =
                        img[1 + (c_im * height + h_pad) * width + w_pad]
                else
                    @inbounds col[1 + (c*height_col+h) * width_col + w] = 0
                end
            end
        end
    end
end

function col2im{T}(col::Array{T}, img::Array{T}, width::Int, height::Int, channels::Int,
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

    kernel_w, kernel_h = kernel
    pad_w, pad_h = pad
    stride_w, stride_h = stride

    height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
    width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
    channels_col = channels * kernel_h * kernel_w

    fill!(img, 0)
    for c = 0:channels_col-1
        w_offset = c % kernel_w
        h_offset = div(c, kernel_w) % kernel_h
        c_im = div(c, kernel_w * kernel_h)
        for h = 0:height_col-1
            for w = 0:width_col-1
                h_pad = h * stride_h - pad_h + h_offset
                w_pad = w * stride_w - pad_w + w_offset
                if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
                    @inbounds img[1 + (c_im * height + h_pad) * width + w_pad] +=
                        col[1 + (c * height_col + h) * width_col + w]
                end
            end
        end
    end
end
