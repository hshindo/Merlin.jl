export Conv1D
export conv1d

doc"""
    Conv1D(T, ksize, insize, outsize, pad, stride, [dilation=1, init_W=Xavier(), init_b=Fill(0)])

1-dimensional convolution function.

```julia
T = Float32
x = Var(rand(T,10,5))
f = Conv1D(T, 5, 10, 3, 2, 1)
y = f(x)
```
"""
mutable struct Conv1D
    W::Var
    b::Var
    ksize::Int
    pad::Int
    stride::Int
    dilation::Int
end

function Conv1D(::Type{T}, ksize::Int, insize::Int, outsize::Int, pad::Int, stride::Int;
    dilation=1, init_W=Xavier(), init_b=Fill(0)) where T

    W = init_W(T, ksize*insize, outsize)
    b = init_b(T, outsize)
    Conv1D(zerograd(W), zerograd(b), ksize, pad, stride, dilation)
end
(f::Conv1D)(args...) = conv1d(args..., f.W, f.b, f.ksize, f.pad, f.stride, f.dilation)

function conv1d(x::Var, batchdims::Vector{Int}, W::Var, b::Var, ksize::Int, pad::Int, stride::Int, dilation::Int)
    if isvoid(x.data)
        h = nothing
        y = nothing
    else
        @assert sum(batchdims) == size(x)[end]
        batchdims_y = map(batchdims) do d
            (d + 2pad - ksize) ÷ stride + 1
        end
        h = window1d(x.data, batchdims_y, ksize, pad, stride, dilation)
        y = linear(h, W.data, b.data)
    end
    Var(y, (conv1d,x,batchdims,W,b,ksize,pad,stride,dilation,h))
end

function addgrad!(y::Var, ::typeof(conv1d), x::Var, batchdims, W::Var, b::Var, ksize, pad, stride, dilation, h)
    gh = zeros(h)
    addgrad!(y, linear, Var(h,grad=gh), W, b)
    isvoid(x.grad) || ∇window1d!(gh, x.grad, batchdims, ksize, pad, stride, dilation)
end

function window1d(x::Matrix{T}, batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int) where T
    y = zeros(T, size(x,1)*ksize, sum(batchdims))
    yi = 1
    s = 1
    for dim in batchdims
        i = s - pad
        while i + ksize <= s + dim + pad
            for w = 0:ksize-1
                j = i + w * dilation
                if j >= s && j < s + dim
                    xi = (j-1) * size(x,1) + 1
                    for c = 0:size(x,1)-1
                        y[yi+c] = x[xi+c]
                    end
                end
                yi += size(x,1)
            end
            i += stride
        end
        s += dim
    end
    y
end

function ∇window1d!(gy::Matrix{T}, gx::Matrix{T}, batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int) where T
    yi = 1
    s = 1
    for dim in batchdims
        i = s - pad
        while i + ksize <= s + dim + pad
            for w = 0:ksize-1
                j = i + w * dilation
                if j >= s && j < s + dim
                    xi = (j-1) * size(gx,1) + 1
                    for c = 0:size(gx,1)-1
                        gx[xi+c] += gy[yi+c]
                    end
                end
                yi += size(gx,1)
            end
            i += stride
        end
        s += dim
    end
end

#=
const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_f32)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_f32_grad)

window2d_handle(::Type{Float32}) = WINDOW2D_F32
∇window2d_handle(::Type{Float32}) = ∇WINDOW2D_F32

doc"""
    Conv2D(T::Type, dims::Tuple, [pads], [strides])

N-dimensional convolution function.

* dims: tuple of ``k1, k2, ..., c_{in}, c_{out}``
where ``k1``, ``k2``, ... are kernel size.
``c_{in}`` and ``c_{out}`` are input channel and output channel, respectively.
* [pads=(0,0,...)]: spatial padding size.
* [strides=(1,1,...)]: kernel strides.

```julia
T = Float32
x = Var(rand(T,5,4,3,2))
f = Conv(T, (2,2,3,4), pads=(0,0), strides=(1,1))
y = f(x)
```
```julia
w = zerograd(uniform(T,-0.01,0.01,2,2,3,4))
b = zerograd(zeros(T,4))
f = Conv(w, b, pads=(0,0), strides=(1,1))
y = f(x)
```
"""
type Conv2D
    w::Var
    b::Var
    insize::NTuple{2,Int}
    outsize::Int
    pad::NTuple{2,Int}
    stride::NTuple{2,Int}
end

function Conv2D{T}(::Type{T}, insize::NTuple{2,Int}, outsize::Int, pad::NTuple{2,Int}, stride::NTuple{2,Int})
    w = uniform(T, -0.001, 0.001, outsize, insize)
    b = zeros(T, outsize, 1)
    Conv2D(zerograd(w), zerograd(b), insize, outsize, pad, stride)
end

function (f::Conv2D)(x::Var)
    y = Var(nothing, f, (x,))
    conv2d!(y, x.data, f.insize, f.outsize, f.pad, f.stride)
    y
end
=#
