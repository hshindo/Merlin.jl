export Convolution

const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)
#const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f64)
#const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f64)

"""
N-dimensional convolution function.

## Functions
- `Convolution{T,N}(::Type{T}, filterdims, stridedims, paddims)`
- `nfilters::NTuple{2,Int}`: number of filters
- `filterdims::NTuple{N,Int}`: filter size
- `stridedims::NTuple{N,Int}`: stride size
- `paddims::NTuple{N,Int}`: padding size

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
w = Var(rand(Float32,2,2,3,4))
b = Var()
f = Convolution(Float32, (), (1,1), (1,1))
y = f(x)
```
"""
type Convolution{T,N} <: Functor
  nfilters::Int
  filterdims::NTuple{N,Int}
  stridedims::NTuple{N,Int}
  paddims::NTuple{N,Int}
  w::Var
end

function Convolution{T}(::Type{T}, filterdims, stridedims, paddims)
  l = Linear(T, prod(channeldims), prod(filterdims))
  Convolution{T,N}(filterdims, stridedims, paddims, l.w)
end

fw_handle(f::Convolution{Float32,2}) = WINDOW2D_FWD_F32_HANDLE
#fw_handle(f::Convolution{Float64,2}) = WINDOW2D_FWD_F64_HANDLE
bw_handle(f::Convolution{Float32,2}) = WINDOW2D_BWD_F32_HANDLE
#bw_handle(f::Convolution{Float64,2}) = WINDOW2D_BWD_F64_HANDLE

function forward(f::Convolution, args::Vector{Var})
  x = args[1]
  y = convolution(f, x.val)
  backward! = gy -> begin
    #hasgrad(x) || return
    #∇window(f, params, x.grad, gy)
    #∇conv!(f, params, x.grad, gy)
  end
  Var(y, nothing, f, args, backward!)
end

function outsize{T,N}(f::Convolution{T,N}, x)
  dims = Array(Int, N+2)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.paddims[i] - f.filterdims[i]) ÷ f.stridedims[i] + 1
  end
  dims[N+1] = f.channeldims[2]
  dims[N+2] = size(x, N+2)
  tuple(dims...)
end

"""
Convolution based on GEMM
"""
function conv_test2{T}(x::Array{T}, y::Array{T}, sizes::Vector{Cint})
  #h = IM2COL_FWD_F32_HANDLE
  h = IM2COL_FWD_F32_HANDLE
  ccall(h, Void,
    (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
    x, y, sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6],
    sizes[7], sizes[8], sizes[9])
  y
end
function conv_test{T}(x::Array{T}, y::Array{T}, sizes::Vector{Cint})
  h = WINDOW2D_FWD_F32_HANDLE
  ccall(h, Void,
    (Ptr{T}, Ptr{T}, Ptr{Cint}),
    x, y, sizes)
  y
end

function convolution{T,N}(f::Convolution{T,N}, x::Array{T})
  y = similar(x, 288)
  h = fw_handle(f)
  s = size(x)
  sizes = Cint[s[1:end-2]..., s[end-1]*s[end], f.filterdims..., f.stridedims..., f.paddims...]
  ccall(h, Void,
    (Ptr{T}, Ptr{T}, Ptr{Cint}),
    x, y, sizes)
  y
end

function convolution{T,N}(f::Convolution{T,N}, x::CuArray{T}, w::CuArray{T})
  CUDNN.convolution!(x, w, f.paddims, f.stridedims, similar(x, outsize(f,x)...))
end

function window{T}(f::Convolution{T,2}, params::Vector{Cint}, x::Array{T})
  n1 = (size(x,1) + 2*pd[1] - fd[1]) ÷ sd[1] + 1
  n2 = (size(x,2) + 2*pd[2] - fd[2]) ÷ sd[2] + 1
  y = Array(T, fd[1]*fd[2], n1*n2)
  ccall(fw_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Cint, Cint),
    x, params, y, size(x,1), size(x,2))
  y
end

function ∇window{T}(f::Convolution{T,2}, params::Vector{Cint}, x::Array{T}, gy::Array{T})
  gx = zeros(x)
  ccall(bw_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(gx,1), size(gx,2))
  gx
end

function aaaaa(f::Convolution)
  T = "float"
  cpp = """
  void window2d_fwd($T *x, $T *y) {
    int x1 = $(x1), x2 = $(x2), x3 = $(x3);
    int w1 = $(w1), w2 = $(w2);
    int s1 = $(s1), s2 = $(s2);
    int p1 = $(p1), p2 = $(p2);
    int n1 = (x1 + 2 * p1 - w1) / s1 + 1;
    int n2 = (x2 + 2 * p2 - w2) / s2 + 1;
    int o = 0;
    for (int i = 0; i < w1 * w2 * x3; i++) {
      int d1 = i % w1;
      int d2 = (i / w1) % w2;
      int d3 = i / (w1 * w2);

      for (int k2 = 0; k2 < n2; k2++) {
        for (int k1 = 0; k1 < n1; k1++) {
          int i1 = k1*s1 - p1 + d1;
          int i2 = k2*s2 - p2 + d2;
          if (i1 >= 0 && i1 < x1 && i2 >= 0 && i2 < x2) {
            y[o] = x[i1 + x1*i2 + x1*x2*d3];
          }
          else y[o] = 0;
          o++;
        }
      }
    }
  }"""
  cpp
end
