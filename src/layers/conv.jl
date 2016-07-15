export Conv
import Base.conv

const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_float)
const WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_double)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_grad_float)
const ∇WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_grad_double)

"""
    Conv(w, [stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight (windowsize, input channel, output channel)
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* padsize::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Data(rand(Float32,5,4,3,2))
f = Conv(Param(rand(Float32,2,2,3,4)), stride=(1,1), padsize=(0,0))
y = f(x)
```
"""
type Conv{N} <: Var
  data
  grad
  tails::Vector
  winsize::NTuple{N,Int}
  stride::NTuple{N,Int}
  padsize::NTuple{N,Int}
  work
end

function Conv(w::Var; stride=(), padsize=())
  N = ndims(w.data) - 2
  winsize = tuple([size(w.data,i) for i=1:N]...)
  length(stride) == 0 && (stride = ntuple(_ -> 1, N))
  length(padsize) == 0 && (padsize = ntuple(_ -> 0, N))
  Conv(nothing, nothing, [w], winsize, stride, padsize, nothing)
end

@compat function (f::Conv{N}){N}(w::Var, x::Var)
  (hasdata(w) && hasdata(x)) || return Conv(nothing, nothing, [w,x], f.winsize, f.stride, f.padsize, nothing)
  y = Conv(nothing, nothing, [w,x], f.winsize, f.stride, f.padsize, nothing)
  conv!(y, w.data, x.data)
end
@compat (y::Conv{N}){N}(x::Var) = y(y[1], x)

handle(::Type{Conv{2}}, ::Type{Float32}) = WINDOW2D_F32
handle(::Type{Conv{2}}, ::Type{Float64}) = WINDOW2D_F64
∇handle(::Type{Conv{2}}, ::Type{Float32}) = ∇WINDOW2D_F32
∇handle(::Type{Conv{2}}, ::Type{Float64}) = ∇WINDOW2D_F64

function conv!{T,N}(out::Conv{N}, w::Array{T}, x::Array{T})
  @assert ndims(w) == ndims(x) == N+2
  work = window(out, x)
  w = reshape(w, size(work,2), size(w,N+2))
  y = Array(T, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,N+2)
    BLAS.gemm!('N', 'N', T(1), view(work,:,:,i), w, T(0), view(y,:,:,i))
  end
  out.work = work
  out.data = reshape(y, outsize(out,x)..., size(y,2), size(y,3))
  out
end

function conv{T}(f::Conv, w::CuArray{T}, x::CuArray{T})
  desc = ConvolutionDesc(T, f.pad, f.stride)
  convolution(x, w, desc)
end

function backward!{N}(out::Conv{N})
  # CPU
  w, x = out[1], out[2]
  for i = 1:size(out.data,N+2)
    ∇linear!(Data())
    hasgrad(w) && BLAS.gemm!('N', 'T', T(1), out.grad, x.data, T(1), w.grad)
    hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.data, v.grad, T(1), x.grad)

    BLAS.gemm!('N', 'N', T(1), view(work,:,:,i), w, T(0), view(y,:,:,i))
  end
  gwork = ∇window(out, out.grad)


  # GPU
  #convdesc = ConvolutionDesc(T, f.pad, f.stride)
  #isempty(gw) || ∇convolution_filter!(x, gy, convdesc, gw)
  #isempty(gx) || ∇convolution_data!(w, gy, convdesc, gx)
end

function window{T,N}(out::Conv{N}, x::Array{T})
  h = handle(Conv{N}, T)
  y = Array(T, prod(outsize(out,x)), prod(out.winsize)*size(x,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    x, y, xsize, Cint[out.winsize...], Cint[out.stride...], Cint[out.padsize...])
  y
end

function outsize{N}(out::Conv{N}, x::Array)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*out.padsize[i] - out.winsize[i]) ÷ out.stride[i] + 1
  end
  dims
end

function ∇window{T,N}(out::Conv{N}, gy::Array{T})
  gx = zeros(out.work)
  h = ∇handle(Conv{N}, T)
  xsize = Cint[size(gx,i) for i=1:N+1]
  xsize[N+1] *= size(gx, N+2)
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    gx, gy, xsize, Cint[out.winsize...], Cint[out.stride...], Cint[out.padsize...])
  gx
end
