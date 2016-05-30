export convolution

const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)
#const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f64)
#const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f64)
#fw_handle(f::Convolution{Float32,2}) = WINDOW2D_FWD_F32_HANDLE
#fw_handle(f::Convolution{Float64,2}) = WINDOW2D_FWD_F64_HANDLE
#bw_handle(f::Convolution{Float32,2}) = WINDOW2D_BWD_F32_HANDLE
#bw_handle(f::Convolution{Float64,2}) = WINDOW2D_BWD_F64_HANDLE

"""
Convolution function.

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
f = Convolution((2,2,3,4),(1,1),(0,0))
y = f(x)
```
"""
type Convolution{N} <: Functor
  w::Var
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end
@compat (f::Convolution)(x::Var) = forward0(f, [x])

function Convolution{T,N}(::Type{T}, weight, stride, pad)
  Convolution(rand(T, weight), stride, pad)
end

function forward(f::Convolution, x::Var)
  convolution(x.val, w.val, stride, pad)
end

function convolution{T}(x::Array{T,4}, w::Array{T,4}, stride, pad)
  h = WINDOW2D_FWD_F32_HANDLE
  sizex = Cint[size(x,1), size(x,2), size(x,3)*size(x,4)]
  params = Cint[size(w,1), size(w,2), stride..., pad...]

  dims = Array(Int, 2)
  for i = 1:length(dims)
    dims[i] = (size(x,i) + 2*pad[i] - size(w,i)) ÷ stride[i] + 1
  end
  work = similar(x, prod(dims)*size(w,3),size(x,4))
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, work, sizex, params)
  work
end
