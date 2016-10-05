export Conv
import Base.conv

const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const IM2COL_F64 = Libdl.dlsym(libmerlin, :im2col_f64)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)
const ∇IM2COL_F64 = Libdl.dlsym(libmerlin, :im2col_f64_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
im2col_handle(::Type{Float64}) = IM2COL_F64
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32
∇im2col_handle(::Type{Float64}) = ∇IM2COL_F64

"""
    Conv(T, channel, filter, [stride, pad])

N-dimensional convolution function.

* T: Type
* filterdims::NTuple{N,Int}: window size
* channeldims::Tuple{Int,Int}: input channel, output channel
* [stride::NTuple{N,Int}]: stride size. Default: (1,1,...)
* [paddims::NTuple{N,Int}]: padding size. Default: (0,0,...)

```julia
x = Var(rand(Float32,5,4,3,2))
f = Conv(Float32, (2,2), (3,4), stride=(1,1), paddims=(0,0))
y = f(x)
```
"""
type Conv{N} <: Functor
    w::Var
    filterdims::NTuple{N,Int}
    stride::NTuple{N,Int}
    paddims::NTuple{N,Int}
end

function Conv(T::Type, filterdims, channeldims; stride=(), paddims=())
    N = length(filterdims)
    w = Var(rand(T(-0.001), T(0.001), filterdims..., channeldims...))
    length(stride) == 0 && (stride = ntuple(_ -> 1, N))
    length(paddims) == 0 && (paddims = ntuple(_ -> 0, N))
    Conv(w, filterdims, stride, paddims)
end

function (f::Conv)(x::Var)
    y, work = conv(f.w.data, x.data, f.filterdims, f.stride, f.paddims)
    function df(gy)
        ∇conv!(f.w.data, f.w.grad, x.data, x.grad, work, y, gy,
        f.filterdims, f.stride, f.paddims)
    end
    Var(y, [x], conv, df)
end

function conv{T}(w::Array{T}, x::Array{T}, windims, stride, paddims)
    N = length(windims)
    outdims = outsize(x, windims, stride, paddims)
    work = Array(T, prod(outdims), prod(windims)*size(x,N+1), size(x,N+2))
    im2col!(x, work, windims, stride, paddims)

    w = reshape(w, size(work,2), size(w,N+2))
    y = gemm(work, w)
    y = reshape(y, outdims..., size(y,2), size(y,3))
    y, work
end

conv(w::CuArray, x::CuArray, windims, stride, paddims) = convolution(x, w, paddims, stride)

function im2col!{T}(x::Array{T}, y::Array{T}, windims, stride, paddims)
    N = length(windims)
    h = im2col_handle(T)
    xsize = Cint[size(x,i) for i=1:N+1]
    xsize[N+1] *= size(x, N+2)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    x, y, xsize, Cint[windims...], Cint[stride...], Cint[paddims...])
    y
end

function outsize(x::Array, windims, stride, paddims)
    N = length(windims)
    Int[(size(x,i)+2*paddims[i]-windims[i]) ÷ stride[i] + 1 for i=1:N]
end

function ∇conv!(w::Array, gw, x::Array, gx, work::Array, y::Array, gy::Array,
    windims, stride, paddims)

    N = length(windims)
    w = reshape(w, size(work,2), size(w,ndims(w)))
    gw == nothing || (gw = reshape(gw, size(work,2), size(gw,ndims(gw))))
    gwork = zeros(work)
    y = reshape(y, size(y,1)*size(y,2), size(y,3), size(y,4))
    gy = reshape(gy, size(gy,1)*size(gy,2), size(gy,3), size(gy,4))
    ∇gemm!(work, gwork, w, gw, y, gy)
    gx == nothing || ∇im2col!(gx, gwork, windims, stride, paddims)
end

function ∇conv!(w::CuArray, gw::CuArray, x::CuArray, gx::CuArray,
    work::CuArray, y::CuArray, gy::CuArray, windims, stride, paddims)

    isempty(gw) || ∇convolution_filter!(x, gy, paddims, stride, gw)
    isempty(gx) || ∇convolution_data!(x, w, gy, paddims, stride, gx)
end

function ∇im2col!{T}(gx::Array{T}, gy::Array{T}, windims, stride, paddims)
    N = length(windims)
    h = ∇im2col_handle(T)
    xsize = Cint[size(gx,i) for i=1:N+1]
    xsize[N+1] *= size(gx, N+2)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    gx, gy, xsize, Cint[windims...], Cint[stride...], Cint[paddims...])
    gx
end

function update!(f::Conv, opt)
    opt(f.w.data, f.w.grad)
end
