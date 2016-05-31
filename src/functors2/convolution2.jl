export Convolution

"""
N-dimensional convolution function.

## Functions
- `Convolution{T,N}(::Type{T}, filterdims, stridedims, paddims)`
- `nfilters::Tuple{Int,Int}`: number of input and output filters.
- `filterdims::NTuple{N,Int}`: filter size.
- `stridedims::NTuple{N,Int}`: stride size (default value: 1).
- `paddims::NTuple{N,Int}`: padding size (default value: 0).

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
f = Convolution(Float32, (3,4), (2,2)) # 2-d convolution
y = f(x) # output size: (4,3,4,2)
```
"""
type Convolution{T,N} <: Functor
  w::Var
  channels::Tuple{Int,Int}
  filterdims::NTuple{N,Int}
  stridedims::NTuple{N,Int}
  paddims::NTuple{N,Int}
  w::Var
end

function Convolution{T,N}(::Type{T}, channels::NTuple{N,Int}, filterdims, stridedims, paddims)
  l = Linear(T, prod(nfilters), prod(filterdims))
  Convolution{T,N}(nfilters, filterdims, stridedims, paddims, l)
end

@compat (f::Convolution)(x::Var) = forward0(f, [x])

const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)
#const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f64)
#const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f64)
fw_handle(f::Convolution{Float32,2}) = WINDOW2D_FWD_F32_HANDLE
#fw_handle(f::Convolution{Float64,2}) = WINDOW2D_FWD_F64_HANDLE
bw_handle(f::Convolution{Float32,2}) = WINDOW2D_BWD_F32_HANDLE
#bw_handle(f::Convolution{Float64,2}) = WINDOW2D_BWD_F64_HANDLE

function forward(f::Convolution, args::Vector{Var})
  x = args[1]
  y = convolution(f, x.val)
  backward! = gy -> begin
    hasgrad(x) || return
    #∇window(f, params, x.grad, gy)
    #∇conv!(f, params, x.grad, gy)
  end
  Var(y, nothing, f, args, backward!)
end

function outsize{T,N}(f::Convolution{T,N}, x)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.paddims[i] - f.filterdims[i]) ÷ f.stridedims[i] + 1
  end
  dims
end

function convolution{T}(f::Convolution{T,2}, x::Array{T,4})
  n = size(x,4)
  o = outsize(f, x)
  work = similar(x, prod(o)*prod(f.filterdims)*f.nfilters[1]*size(x,4))
  sizex = Cint[size(x,1), size(x,2), size(x,3)*size(x,4)]
  params = Cint[f.filterdims..., f.stridedims...,f.paddims...]
  ccall(fw_handle(f), Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, work, sizex, params)

  y = similar(x, o..., f.nfilters[2], size(x,4))
  u = length(work) / n
  u2 =
  for i = 1:size(x,4)
    ww = pointer_to_array(pointer(work,i*u),u)
    yy = pointer_to_array(pointer(y,i*u),u)
    gemm!('N', 'N', T(1), ww, f.w.val, T(0), yy)
  end
end

function convolution{T,N}(f::Convolution{T,N}, x::CuArray{T}, w::CuArray{T})
  CUDNN.convolution!(x, w, f.paddims, f.stridedims, similar(x, outsize(f,x)...))
end

function ∇window{T}(f::Convolution{T,2}, params::Vector{Cint}, x::Array{T}, gy::Array{T})
  gx = zeros(x)
  ccall(bw_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(gx,1), size(gx,2))
  gx
end

#=
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
=#
