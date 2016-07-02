export Conv
import Base.conv

const WINDOW2D_FWD_F32 = Libdl.dlsym(libmerlin, :window2d_fwd_f32)
const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_f32)
#const WINDOW2D_BWD_F32 = Libdl.dlsym(library, :window2d_bwd_f32)
#const WINDOW2D_FWD_F64 = Libdl.dlsym(library, :window2d_fwd_f64)
#const WINDOW2D_BWD_F64 = Libdl.dlsym(library, :window2d_bwd_f64)

function window2{T,N}(stride::NTuple{N,Int}, w::Array{T}, x::Array{T})
  outdims = Array(Int, N)
  for i = 1:N
    outdims[i] = (size(x,i) - size(w,i)) ÷ stride[i] + 1
  end
  winsize = [size(w,i) for i=1:N]
  h = WINDOW2D_FWD_F32
  y = similar(x, prod(outdims), prod(winsize)*size(w,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[winsize..., stride..., 0, 0]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

function window3{T,N}(stride::NTuple{N,Int}, w::Array{T}, x::Array{T})
  outdims = Array(Int, N)
  for i = 1:N
    outdims[i] = (size(x,i) - size(w,i)) ÷ stride[i] + 1
  end
  winsize = [size(w,i) for i=1:N]
  h = WINDOW2D_F32
  y = similar(x, prod(outdims), prod(winsize)*size(w,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[winsize..., stride..., 0, 0]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

"""
    Conv(w, [stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* pad::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
f = Conv(rand(Float32,2,2,3,4), stride=(1,1), pad=(0,0))
y = f(x)
```
"""
type Conv{N}
  w::Var
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

function Conv(w::Var; stride=(), pad=())
  N = ndims(w.value) - 2
  length(stride) == 0 && (stride = ntuple(_->1, N))
  length(pad) == 0 && (pad = ntuple(_->0, N))
  Conv(w, stride, pad)
end

@compat (f::Conv)(x::Var) = ConvFun(f.stride,f.pad)(f.w, x)


type ConvFun{N}
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

handle(::ConvFun{2}, ::Type{Float32}) = WINDOW2D_FWD_F32, WINDOW2D_BWD_F32
handle(::ConvFun{2}, ::Type{Float64}) = WINDOW2D_FWD_F64, WINDOW2D_BWD_F64

@compat function (f::ConvFun{N}){N}(w::Var, x::Var)
  @checkargs f (w,x)
  y = conv(f, w.value, x.value)
  df(gy) = begin
    gwork = zeros(work)
    ∇window!(f, gy)
  end
  Var(y, df, [w,x])
end

# GEMM-based convolution
function conv{T,N}(f::ConvFun{N}, w::Array{T}, x::Array{T})
  @assert ndims(w) == ndims(x) == N+2
  # x -> work
  outdims = outsize(f, w, x)
  work = window(f, w, x, outdims)

  # work -> y
  w = reshape(w, size(work,2), size(w,N+2))
  y = similar(x, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,N+2)
    BLAS.gemm!('N', 'N', T(1), slice(work,:,:,i), w, T(0), slice(y,:,:,i))
  end
  y = reshape(y, outdims..., size(y,2), size(y,3))
  y
end

function conv{T}(f::ConvFun, w::CuArray{T}, x::CuArray{T})
  desc = ConvolutionDesc(T, f.pad, f.stride)
  convolution(x, w, desc)
end

function ∇conv!{T,N}(f::ConvFun{N}, gx::Array{T}, gy::Array{T})
  for i = 1:size(x,N+2)
    BLAS.gemm!('N', 'T', T(1), gy, x2.value, T(1), x1.grad)
    BLAS.gemm!('T', 'N', T(1), x1.value, gy, T(1), x2.grad)
  end

  outdims = outsize(f, gw, gx)
  ∇window!(f, w, gx, gy, outdims)
end

function backward!{T,N}(f::ConvFun{N}, w::Array{T}, x::Array{T},
  gw::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})

  work, gwork = f.work, zeros(f.work)
  for i = 1:size(x,N+2)
    ∇times!(slice(work), w, slice(gwork), gw, slice(y), slice(gy))
  end
  col2im!(f, gx, gwork)
end

function backward!{T}(f::ConvFun, x, gx, y, gy::CuArray{T})
  convdesc = ConvolutionDesc(T, f.pad, f.stride)
  isempty(gw) || ∇convolution_filter!(x, gy, convdesc, gw)
  isempty(gx) || ∇convolution_data!(w, gy, convdesc, gx)
end

function outsize{N}(f::ConvFun{N}, w::Array, x::Array)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.pad[i] - size(w,i)) ÷ f.stride[i] + 1
  end
  dims
end

function window{T,N}(f::ConvFun{N}, w::Array{T}, x::Array{T}, outdims::Vector{Int})
  winsize = [size(w,i) for i=1:N]
  h = handle(f,T)[1]
  y = similar(x, prod(outdims), prod(winsize)*size(w,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[winsize..., f.stride..., f.pad...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

function ∇window!{T,N}(f::ConvFun{N}, w::Array{T}, x::Array{T}, gy::Array{T}, outdims::Vector{Int})
  gx = zeros(x)
  winsize = [size(w,i) for i=1:N]
  h = handle(f,T)[2]
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[winsize..., f.stride..., f.pad...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), gx, gy, xsize, params)
  gx
end
