export Convolution

const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)
const WINDOW2D_FWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f64)
const WINDOW2D_BWD_F64_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f64)

"""
N-dimensional convolution function.
The parameters are: filter \$w\$, and the bias \$b\$.

Notation:

    - \$c_{in}\$ and \$c_{out}\$: number of the input and output, respectively.
    - \$h\$ and \$w\$: the height and width of the input image, respectively.
    - \$k_h\$ and \$k_w\$: the height and width of the filters, respectively.

## Functions
- `Convolution{N}(w, b, strides, pads)`
- `w::Var`: weight variable of shape \$(c_{out}, c_{in}, k_w, k_h)\$
- `b::Var`: bias variable
- `strides::NTuple{N,Int}`
- `pads::NTuple{N,Int}`

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
w = Var(rand(Float32,2,2,3,4))
b = Var()
f = Convolution(Float32, w, b, (1,1), (1,1))
y = f(x)
```
"""
type Convolution{N} <: Functor
  w::Var
  b::Var
  strides::NTuple{N,Int}
  pads::NTuple{N,Int}
end

function Convolution{T}(::Type{T}, filters, strides, pads)
  n = filters[1]
  l = Linear(T, n, prod(filters))
  Convolution(l.w, l.b, strides, pads)
end

fw_handle(f::Convolution{2}, ::Type{Float32}) = WINDOW2D_FWD_F32_HANDLE
fw_handle(f::Convolution{2}, ::Type{Float64}) = WINDOW2D_FWD_F64_HANDLE
bw_handle(f::Convolution{2}, ::Type{Float32}) = WINDOW2D_BWD_F32_HANDLE
bw_handle(f::Convolution{2}, ::Type{Float64}) = WINDOW2D_BWD_F64_HANDLE

function forward(f::Convolution{2}, args::Vector{Var})
  x = args[1]
  fd, sd, pd = f.filter_dims, f.stride_dims, f.pad_dims
  params = Cint[fd..., sd..., pd...]
  w = window(f, params, x.val)
  y = linear(Var(w))

  backward! = gy -> begin
    hasgrad(x) || return
    ∇window(f, params, x.grad, gy)
    ∇conv!(f, params, x.grad, gy)
  end
  Var(y, nothing, f, args, backward!)
end

function window{T}(f::Convolution{2}, params::Vector{Cint}, x::Array{T})
  n1 = (size(x,1) + 2*pd[1] - fd[1]) ÷ sd[1] + 1
  n2 = (size(x,2) + 2*pd[2] - fd[2]) ÷ sd[2] + 1
  y = Array(T, fd[1]*fd[2], n1*n2)
  ccall(fw_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Cint, Cint),
    x, params, y, size(x,1), size(x,2))
  y
end

function ∇window{T}(f::Convolution{2}, params::Vector{Cint}, x::Array{T}, gy::Array{T})
  gx = zeros(x)
  ccall(bw_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(gx,1), size(gx,2))
  gx
end
