export Conv
import Base.conv

const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_float)
const WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_double)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_grad_float)
const ∇WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_grad_double)

type Conv{N} <: Var
    data
    grad
    tails::Vector
    windims::NTuple{N,Int}
    stride::NTuple{N,Int}
    paddims::NTuple{N,Int}
    work
end

handle(::Type{Conv{2}}, ::Type{Float32}) = WINDOW2D_F32
handle(::Type{Conv{2}}, ::Type{Float64}) = WINDOW2D_F64
∇handle(::Type{Conv{2}}, ::Type{Float32}) = ∇WINDOW2D_F32
∇handle(::Type{Conv{2}}, ::Type{Float64}) = ∇WINDOW2D_F64

"""
    Conv(w, [stride, pad])

N-dimensional convolution function.

## Arguments
* w: weight (windowsize, input channel, output channel)
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* paddims::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Data(rand(Float32,5,4,3,2))
f = Conv(rand(Float32,2,2,3,4), stride=(1,1), padsize=(0,0))
y = f(x)
```
"""
function Conv(w::Var; stride=(), paddims=())
    N = ndims(w.data) - 2
    windims = tuple([size(w.data,i) for i=1:N]...)
    length(stride) == 0 && (stride = ntuple(_ -> 1, N))
    length(paddims) == 0 && (paddims = ntuple(_ -> 0, N))
    Conv(nothing, nothing, [w], windims, stride, paddims, nothing)
end
Conv(w::Array; stride=(), paddims=()) = Conv(Param(w), stride=stride, paddims=paddims)

@compat function (c::Conv{N}){N}(w::Var, x::Var)
    windims, stride, paddims = c.windims, c.stride, c.paddims
    hasdata(w,x) || return Conv(nothing, nothing, [w,x], windims, stride, paddims, nothing)
    y, work = conv(w.data, x.data, windims, stride, paddims)
    Conv(y, nothing, [w,x], windims, stride, paddims, work)
end
@compat (c::Conv)(x::Var) = c(c[1], x)

function conv{T}(w::Array{T}, x::Array{T}, windims, stride, paddims)
    N = length(windims)
    @assert ndims(w) == ndims(x) == N+2

    outdims = outsize(x, windims, stride, paddims)
    work = Array(T, prod(outdims), prod(windims)*size(x,N+1), size(x,N+2))
    window!(x, work, windims, stride, paddims)

    w = reshape(w, size(work,2), size(w,N+2))
    y = gemm(work, w)
    #y = Array(T, size(work,1), size(w,2), size(work,3))
    #for i = 1:size(x,N+2)
    #    BLAS.gemm!('N', 'N', T(1), view(work,:,:,i), w, T(0), view(y,:,:,i))
    #end
    y = reshape(y, outdims..., size(y,2), size(y,3))
    y, work
end

function window!{T}(x::Array{T}, y::Array{T}, windims, stride, paddims)
    N = length(windims)
    h = handle(Conv{N}, T)
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

function backward!(y::Conv)
    w, x = y[1], y[2]
    ∇conv!(w.data, w.grad, x.data, x.grad, y.work, y.data, y.grad,
    y.windims, y.stride, y.paddims)
end

function ∇conv!(w::Array, gw, x::Array, gx, work::Array, y::Array, gy::Array,
    windims, stride, paddims)

    w = reshape(w, size(work,2), size(w,ndims(w)))
    gw == nothing || (gw = reshape(gw, size(work,2), size(gw,ndims(gw))))
    gwork = zeros(work)
    y = reshape(y, size(y,1)*size(y,2), size(y,3), size(y,4))
    gy = reshape(gy, size(gy,1)*size(gy,2), size(gy,3), size(gy,4))
    ∇gemm!(work, gwork, w, gw, y, gy)
    gx == nothing || ∇window!(gx, gwork, windims, stride, paddims)
end

function ∇conv!(w::CuArray)
    #convdesc = ConvolutionDesc(T, f.pad, f.stride)
    #isempty(gw) || ∇convolution_filter!(x, gy, convdesc, gw)
    #isempty(gx) || ∇convolution_data!(w, gy, convdesc, gx)
end

function ∇window!{T}(gx::Array{T}, gy::Array{T}, windims, stride, paddims)
    N = length(windims)
    h = ∇handle(Conv{N}, T)
    xsize = Cint[size(gx,i) for i=1:N+1]
    xsize[N+1] *= size(gx, N+2)
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    gx, gy, xsize, Cint[windims...], Cint[stride...], Cint[paddims...])
    gx
end
