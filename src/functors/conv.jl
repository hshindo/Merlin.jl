export Conv

const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f64)

"""
## Conv

Convolution function.

- `Conv(w, strides::NTuple{N,Int}, pads::NTuple{N,Int})`

### 👉 Example
```julia
x = Var(rand(Float32,50,10))
f = Conv(5,(10,2),(10,1),(5,5))
y = f(x)
```
"""
type Conv{N} <: Functor
  num_filter::Int
  filter_dims::NTuple{N,Int}
  stride_dims::NTuple{N,Int}
  pad_dims::NTuple{N,Int}
  w::Var
  b::Var
end

function Conv{T}(::Type{T}, num_filter, filter_dims, stride_dims, pad_dims)
  l = Linear(T, num_filter, prod(filter_dims))
  Conv(num_filter, filter_dims, stride_dims, pad_dims, l.w, l.b)
end

@compat (f::Conv)(x) = forward(f, x)

fw_handle(f::Conv{2}, ::Type{Float32}) = WINDOW2D_FWD_F32_HANDLE
fw_handle(f::Conv{2}, ::Type{Float64}) = WINDOW2D_FWD_F64_HANDLE
bw_handle(f::Conv{2}, ::Type{Float32}) = WINDOW2D_BWD_F32_HANDLE
bw_handle(f::Conv{2}, ::Type{Float64}) = WINDOW2D_BWD_F64_HANDLE

function forward(f::Conv{2}, x::Var)
  fd, sd, pd = f.filter_dims, f.stride_dims, f.pad_dims
  params = Cint[fd..., sd..., pd...]
  n1 = (size(x,1) + 2*pd[1] - fd[1]) ÷ sd[1] + 1
  n2 = (size(x,2) + 2*pd[2] - fd[2]) ÷ sd[2] + 1
  cols = Array(T, fd[1]*fd[2], n1*n2)
  ccall(fw_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Cint, Cint),
    x, params, cols, size(x,1), size(x,2))
  y = f.w * cols .+ f.b

  backward! = gy -> begin
    hasgrad(x) || return
    gx = x.grad
    ccall(bw_handle(f,T), Void,
      (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
      params, gy, gx, size(gx,1), size(gx,2))
  end
  Var(y, nothing, f, [x], backward!)
end
