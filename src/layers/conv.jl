export Conv
import Base.conv

"""
    Conv(w, [stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* padsize::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Data(rand(Float32,5,4,3,2))
f = Conv(Float32, (2,2,3,4), stride=(1,1), padsize=(0,0))
y = f(x)
```
"""
type Conv{N}
  data
  grad
  tails::Vector
end

function Conv(w::Var; stride=(), padsize=())
  N = ndims(w.value) - 2
  winsize = [size(f.w.value,i) for i=1:N]
  length(stride) == 0 && (stride = ntuple(_ -> 1, N))
  length(padsize) == 0 && (padsize = ntuple(_ -> 0, N))
  Conv(nothing, nothing, [w], winsize, stride, padsize, nothing)
end

@compat function (f::Conv{N}){N}(w::Var, x::Var)
  (hasdata(w) && hasdata(x)) || return Conv(nothing, nothing, [w,x], f.stride, f.padsize)
  y, work = conv(f.winsize, f.stride, f.padsize, f[1], f[2])
  Conv(y, nothing, [w,x], f.stride, f.padsize, work)
end
@compat (f::Conv{N}){N}(x::Var) = f(f[1], x)

function conv{T}(winsize, stride, padsize, w::Array{T}, x::Array{T})
  N = length(winsize)
  @assert ndims(w) == ndims(x) == N+2
  work = window(f, x)
  w = reshape(w, size(work,2), size(w,N+2))
  y = Array(T, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,N+2)
    BLAS.gemm!('N', 'N', T(1), view(work,:,:,i), w, T(0), view(y,:,:,i))
  end
  y = reshape(y, outsize(f,x)..., size(y,2), size(y,3))
  y, work
end

function window{T,N}(v::Conv{N}, x::Array{T})
  h = handle(v, T)
  y = similar(x, prod(outsize(f,x)), prod(f.winsize)*size(x,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[f.winsize..., f.stride..., f.padsize...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

function outsize{N}(f::ConvFun{N}, x)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.padsize[i] - f.winsize[i]) ÷ f.stride[i] + 1
  end
  dims
end

function conv{T}(f::ConvFun, w::CuArray{T}, x::CuArray{T})
  desc = ConvolutionDesc(T, f.pad, f.stride)
  convolution(x, w, desc)
end
















@compat function (f::Conv{N}){N}(x::Var)
  winsize = size(f.w.value)[1:N]
  ConvFun(winsize,f.stride,f.padsize)(f.w,x)
end

type ConvFun{N}
  winsize::NTuple{N,Int}
  stride::NTuple{N,Int}
  padsize::NTuple{N,Int}
end

handle(::ConvFun{2}, ::Type{Float32}) = WINDOW2D_F32, ∇WINDOW2D_F32
handle(::ConvFun{2}, ::Type{Float64}) = WINDOW2D_F64, ∇WINDOW2D_F64

@compat function (f::ConvFun)(w::Var, x::Var)
  @checkargs f (w,x)
  if typeof(w.value) <: Array
    y, work = conv(f, w.value, x.value)
    df(gy) = ∇conv!(f, w, x, work, gy)
  else
    y = conv(f, w.value, x.value)
    df(gy) = ()
  end
  Var(y, df, [w,x])
end

function conv{T,N}(f::ConvFun{N}, w::Array{T}, x::Array{T})
  @assert ndims(w) == ndims(x) == N+2
  work = window(f, x)
  w = reshape(w, size(work,2), size(w,N+2))
  y = Array(T, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,N+2)
    BLAS.gemm!('N', 'N', T(1), slice(work,:,:,i), w, T(0), slice(y,:,:,i))
  end
  y = reshape(y, outsize(f,x)..., size(y,2), size(y,3))
  y, work
end

function conv{T}(f::ConvFun, w::CuArray{T}, x::CuArray{T})
  desc = ConvolutionDesc(T, f.pad, f.stride)
  convolution(x, w, desc)
end

function ∇conv!{T,N}(f::ConvFun{N}, w::Var, x::Var, work::Array{T}, gy::Array{T})
  gy = reshape(gy, Int(length(gy)/size(gy,3)/size(gy,4)), size(gy,3), size(gy,4))
  gwork = similar(work)
  wv = reshape(w.value, size(work,2), size(w.value,N+2))
  gw = reshape(w.grad, size(work,2), size(w.grad,N+2))
  for i = 1:size(work,3)
    gy_i = slice(gy,:,:,i)
    hasgrad(x) && BLAS.gemm!('N', 'T', T(1), gy_i, wv,
      T(0), slice(gwork,:,:,i))
    hasgrad(w) && BLAS.gemm!('T', 'N', T(1), slice(work,:,:,i), gy_i, T(1), gw)
  end
  hasgrad(x) && ∇window!(f, x.grad, gwork)
end

function backward!{T}(f::ConvFun, x, gx, y, gy::CuArray{T})
  convdesc = ConvolutionDesc(T, f.pad, f.stride)
  isempty(gw) || ∇convolution_filter!(x, gy, convdesc, gw)
  isempty(gx) || ∇convolution_data!(w, gy, convdesc, gx)
end

function outsize{N}(f::ConvFun{N}, x)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.padsize[i] - f.winsize[i]) ÷ f.stride[i] + 1
  end
  dims
end

function window{T,N}(f::ConvFun{N}, x::Array{T})
  h = handle(f,T)[1]
  y = similar(x, prod(outsize(f,x)), prod(f.winsize)*size(x,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[f.winsize..., f.stride..., f.padsize...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

function ∇window!{T,N}(f::ConvFun{N}, gx::Array{T}, gy::Array{T})
  h = handle(f,T)[2]
  xsize = Cint[size(gx,i) for i=1:N+1]
  xsize[N+1] *= size(gx, N+2)
  params = Cint[f.winsize..., f.stride..., f.padsize...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), gx, gy, xsize, params)
end
