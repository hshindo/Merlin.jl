export Conv

"""
    Conv(T, filtersize, kwargs...)

* W: (W1,W2,...,I,O)
* X: (X1,X2,...,I,N)
* Y: (Y1,Y2,...,O,N)

where
* I: number of input channels
* O: number of output channels
* N: batch size

```julia
T = Float32
conv = Conv(T, (1,1,3,2))
x = CuArray{T}(5,5,3,3)
y = conv(x)
```
"""
mutable struct Conv{N}
    w::Var
    b::Var
    pads::NTuple{N,Int}
    strides::NTuple{N,Int}
    dilations::NTuple{N,Int}
end

function Conv(::Type{T}, filtersize::Tuple;
    pads=0, strides=1, dilations=1,
    init_w=Xavier(), init_b=Fill(0)) where T

    N = length(filtersize) - 2
    @assert N == 2
    isa(pads,Int) && (pads = ntuple(_->pads,N))
    isa(strides,Int) && (strides = ntuple(_->strides,N))
    isa(dilations,Int) && (dilations = ntuple(_->dilations,N))

    w = init_w(T, filtersize...)
    b = init_b(T, 1)
    Conv(zerograd(w), zerograd(b), pads, strides, dilations)
end

function (conv::Conv)(x::Var)
    y = conv(x.data)
    Var(y, (conv,x))
end

function addgrad!(y::Var, conv::Conv, x::Var)
    ∇conv!(y.grad, conv, x.data, x.grad)
end

function (conv::Conv)(x::CuArray)
    CUDNN.convolution(conv.w.data, x, conv.pads, conv.strides, conv.dilations)
end

function ∇conv!(gy::CuArray, conv::Conv, x::CuArray, gx)
    w, gw = conv.w.data, conv.w.grad
    CUDNN.∇convolution!(gy, w, gw, x, gx, conv.pads, conv.strides, conv.dilations)
end

function compile(conv::Conv, backend)
    w = compile(conv.w, backend)
    b = compile(conv.b, backend)
    Conv(w, b, conv.pads, conv.strides, conv.dilations)
end
