export Conv
import Base.conv

const WINDOW2D_FWD_F32 = Libdl.dlsym(library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32 = Libdl.dlsym(library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64 = Libdl.dlsym(library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64 = Libdl.dlsym(library, :window2d_bwd_f64)

"""
N-d convolution function.
Currently, only 2-d is supported.

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
w = Var(rand(Float32,2,2,3,4))
y = conv(w, x, stride=(1,1), pad=(0,0), bias=true)
```
"""
type Conv{N}
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

handle(::Type{Conv{2}}, ::Type{Float32}) = WINDOW2D_FWD_F32, WINDOW2D_BWD_F32
handle(::Type{Conv{2}}, ::Type{Float64}) = WINDOW2D_FWD_F64, WINDOW2D_BWD_F64

function conv(w::Var, x::Var; stride=(), pad=(), b::Var=Var())
  N = ndims(w.value) - 2
  isempty(stride) && (stride = ntuple(_ -> 1, N))
  isempty(pad) && (pad = ntuple(_ -> 0, N))
  f = Conv(stride, pad)
  args = [w, x]
  b.value == nothing || push!(args, b)
  forward(f, args)
end

function forward{T}(f::Conv{2}, w::Array{T,4}, x::Array{T,4})
  h = handle(Conv{2}, T)[1]
  xsize = Cint[size(x,1), size(x,2), size(x,3)*size(x,4)]
  params = Cint[size(w,1), size(w,2), stride..., pad...]

  dims = Array(Int, 2)
  for i = 1:length(dims)
    dims[i] = (size(x,i) + 2*pad[i] - size(w,i)) ÷ stride[i] + 1
  end
  work = similar(x, prod(dims), size(w,1)*size(w,2)*size(w,3), size(x,4))
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, work, xsize, params)

  w = reshape(w, size(w,1)*size(w,2)*size(w,3), size(w,4))
  y = similar(x, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,4)
    BLAS.gemm!('N', 'N', T(1), slice(work,:,:,i), w, T(0), slice(y,:,:,i))
  end
  y = reshape(y, dims[1], dims[2], size(y,2), size(y,3))
  y, f
end

function forward(f::Conv{2}, w::CuArray, x::CuArray)
  CUDNN.convolution(x, w, f.pad, f.stride)
end
