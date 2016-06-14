export Conv
import Base.conv

const WINDOW2D_FWD_F32 = Libdl.dlsym(library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32 = Libdl.dlsym(library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64 = Libdl.dlsym(library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64 = Libdl.dlsym(library, :window2d_bwd_f64)

type Conv{N} <: Functor
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

"""
    conv(w, x; [stride, pad])

N-dimensional convolution function.

## Arguments
* w::Var: weight variable
* x::Var: input variable
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* pad::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
w = Var(rand(Float32,2,2,3,4))
y = conv(w, x, stride=(1,1), pad=(0,0))
```
"""
function conv(w::Var, x::Var; stride=(), pad=())
  N = ndims(x.value) - 2
  length(stride) == 0 && (stride = ntuple(_->1, N))
  length(pad) == 0 && (pad = ntuple(_->0, N))
  forward(Conv(stride,pad), w, x)
end

handle(::Conv{2}, ::Type{Float32}) = WINDOW2D_FWD_F32, WINDOW2D_BWD_F32
handle(::Conv{2}, ::Type{Float64}) = WINDOW2D_FWD_F64, WINDOW2D_BWD_F64

function outsize{N}(f::Conv{N}, w::Array, x::Array)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.pad[i] - size(w,i)) ÷ f.stride[i] + 1
  end
  dims
end

function im2col{T,N}(f::Conv{N}, w::Array{T}, x::Array{T}, outdims::Vector{Int})
  window = [size(w,i) for i=1:N]
  h = handle(f,T)[1]
  y = similar(x, prod(outdims), prod(window)*size(w,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[window..., f.stride..., f.pad...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

function ∇im2col!{T}(gx::Matrix{T}, gy::Matrix{T})
  filter = [size(w,i) for i=1:N]
  h = handle(Conv{N},T)[2]
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[filter..., stride..., pad...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), gx, gy, xsize, params)
  gx
end

function forward{T,N}(f::Conv{N}, w::Array{T}, x::Array{T})
  outdims = outsize(f, w, x)
  work = im2col(f, w, x, outdims)
  w = reshape(w, size(work,2), size(w,N+2))
  y = similar(x, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,4)
    BLAS.gemm!('N', 'N', T(1), slice(work,:,:,i), w, T(0), slice(y,:,:,i))
  end
  y = reshape(y, outdims..., size(y,2), size(y,3))
  f, y
end

function forward{T}(f::Conv, w::CuArray{T}, x::CuArray{T})
  desc = ConvolutionDesc(T, f.pad, f.stride)
  convolution(x, w, desc)
end

function backward!{T}(f::Conv, w::Array{T}, x::Array{T}, y, gy)

end

function backward!{T}(f::Conv, y, gy::Array{T}, x, gx::Array{T})
  gwork = similar()
  for i = 1:size(x,4)
    gy = slice(gy,:,:,i)
    BLAS.gemm!('N', 'T', T(1), gy, slice(x,:,:,i), T(1), gw)
    BLAS.gemm!('T', 'N', T(1), slice(w), gy, T(1), gwork)
    #isempty(gw) || BLAS.gemm!('N', 'T', T(1), gy, x, T(1), gw)
    #isempty(gx) || BLAS.gemm!('T', 'N', T(1), w, gy, T(1), gx)
  end
  ∇window!(gx, gwork)
end

function backward!{T}(f::Conv, x, gx, y, gy::CuArray{T})
  convdesc = ConvolutionDesc(T, f.pad, f.stride)
  isempty(gw) || ∇convolution_filter!(x, gy, convdesc, gw)
  isempty(gx) || ∇convolution_data!(w, gy, convdesc, gx)
end
