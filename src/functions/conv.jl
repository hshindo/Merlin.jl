export Conv
import Base.conv

const WINDOW2D_FWD_F32 = Libdl.dlsym(library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32 = Libdl.dlsym(library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64 = Libdl.dlsym(library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64 = Libdl.dlsym(library, :window2d_bwd_f64)

"""
    Conv(w, b; [stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight
* b::Var: bias
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* pad::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
f = Conv(rand(Float32,2,2,3,4), stride=(1,1), pad=(0,0))
y = f(x)
```
"""
type Conv
  w::Var
  b::Var
  stride
  pad
end

function Conv(w::Var, b::Var, stride=(), pad=())
  N = ndims(w.value) - 2
  length(stride) == 0 && (stride = ntuple(_->1, N))
  length(pad) == 0 && (pad = ntuple(_->0, N))
  Conv(w, b, stride, pad)
end

@compat (f::Conv)(x::Var) = ConvFun(f.stride,f.pad)(f.w, f.x, f.b)


type ConvFun{N}
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

handle(::ConvFun{2}, ::Type{Float32}) = WINDOW2D_FWD_F32, WINDOW2D_BWD_F32
handle(::ConvFun{2}, ::Type{Float64}) = WINDOW2D_FWD_F64, WINDOW2D_BWD_F64

@compat function (f::ConvFun{N}){N}(w::Var, x::Var, b::Var)
  @checkargs f (w,x,b)
  throw("Not implemented yet.")
  #conv(w.value, f.stride, f.pad, x.value)
end

function conv{T}(w::Array{T}, pad, stride, x::Array{T})
  @assert ndims(w) == ndims(x) == N+2
  outdims = outsize(f, w, x)
  work = im2col(f, w, x, outdims)
  w = reshape(w, size(work,2), size(w,N+2))
  y = similar(x, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,N+2)
    BLAS.gemm!('N', 'N', T(1), slice(work,:,:,i), w, T(0), slice(y,:,:,i))
  end
  y = reshape(y, outdims..., size(y,2), size(y,3))
  f, y
end

function conv{T}(f::ConvFun, w::CuArray{T}, x::CuArray{T})
  desc = ConvolutionDesc(T, f.pad, f.stride)
  f, convolution(x, w, desc)
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

function im2col{T,N}(f::ConvFun{N}, w::Array{T}, x::Array{T}, outdims::Vector{Int})
  window = [size(w,i) for i=1:N]
  h = handle(f,T)[1]
  y = similar(x, prod(outdims), prod(window)*size(w,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[window..., f.stride..., f.pad...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

function col2im!{T,N}(f::ConvFun{N}, gx::Array{T}, gy::Array{T})
  window = [size(w,i) for i=1:N]
  h = handle(f,T)[2]
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[window..., stride..., pad...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), gx, gy, xsize, params)
  gx
end
