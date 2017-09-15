export Conv2D

const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_f32)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_f32_grad)

window2d_handle(::Type{Float32}) = WINDOW2D_F32
∇window2d_handle(::Type{Float32}) = ∇WINDOW2D_F32

"""
    Conv(T::Type, dims::Tuple, [pads], [strides])

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
