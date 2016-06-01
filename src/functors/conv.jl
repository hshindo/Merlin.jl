export Conv
import Base.conv

const WINDOW2D_FWD_F32 = Libdl.dlsym(library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32 = Libdl.dlsym(library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64 = Libdl.dlsym(library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64 = Libdl.dlsym(library, :window2d_bwd_f64)

"""
N-d convolution function.

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
f = Conv((2,2,3,4),(1,1),(0,0))
y = f(x)
```
"""
type Conv{T,N}
  w::Var
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

function Conv(w::Var, stride, pad)
  T = eltype(w.value)
  N = length(stride)
  Conv{T,N}(w, stride, pad)
end

@compat function (f::Conv)(x::Var)

end

function forward{T}(f::Conv, x::Array{T,4})
  h = WINDOW2D_FWD_F32
  w = f.w.value
  xsize = Cint[size(x,1), size(x,2), size(x,3)*size(x,4)]
  params = Cint[size(w,1), size(w,2), f.stride..., f.pad...]

  dims = Array(Int, 2)
  for i = 1:length(dims)
    dims[i] = (size(x,i) + 2*f.pad[i] - size(w,i)) ÷ f.stride[i] + 1
  end
  work = similar(x, prod(dims), size(w,1)*size(w,2)*size(w,3), size(x,4))
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, work, xsize, params)

  w = reshape(w, size(w,1)*size(w,2)*size(w,3), size(w,4))
  y = similar(x, size(work,1), size(w,2), size(work,3))
  for i = 1:size(x,4)
    BLAS.gemm!('N', 'N', T(1), slice(work,:,:,i), w, T(0), slice(y,:,:,i))
  end
  reshape(y, dims[1], dims[2], size(y,2), size(y,3))
end

function conv{T}(f::Conv, x::CuArray{T}, w::CuArray{T})
  dims = Array(Int, 2)
  for i = 1:length(dims)
    dims[i] = (size(x,i) + 2*f.pad[i] - size(w,i)) ÷ f.stride[i] + 1
  end
  push!(dims, size(w,4), size(x,4))
  CUDNN.convolution!(x, w, f.pad, f.stride, similar(x, dims...))
end
