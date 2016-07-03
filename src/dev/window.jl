export window

const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_float)
const WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_double)

"""
    window(x, dims, [stride], [pad])

Extract elements from n-d array along the sliding window.

## Arguments
* `x::Var`: input var
* `dims::NTuple{N,Int}`: window size
* `stride::NTuple{N,Int}`: stride size. Default: 1,1,...
* `pad::NTuple{N,Int}`: padding size. Default: 0,0,...

## 👉 Example
```julia
x = Var(rand(Float32,10,10))
y = window(x)
```
"""
function window{N}(x::Var, dims::NTuple{N,Int}; stride=(), pad=())
  N = ndims(w.value) - 2
  length(stride) == 0 && (stride = ntuple(_->1, N))
  length(pad) == 0 && (pad = ntuple(_->0, N))
  Window(dims, stride, pad)(x)
end

type Window{N}
  dims::NTuple{N,Int}
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

@compat function (f::Window)(x::Var)
  @checkargs (x,)
  y = window(f, x.value, f.dims, f.stride, f.pad)
  df(gy) = hasgrad(x) && ∇window!()
  Var(y, df, [x])
end

handle(f::Window{2}, ::Type{Float32}) = WINDOW2D_FWD_F32, WINDOW2D_BWD_F32
handle(f::Window{2}, ::Type{Float64}) = WINDOW2D_FWD_F64, WINDOW2D_BWD_F64

function window{T,N}(f::Window{N}, x::Array{T})
  outdims = outsize(f, w, x)
  winsize = [size(w,i) for i=1:N]
  h = handle(f,T)[1]
  y = similar(x, prod(outdims), prod(winsize)*size(w,N+1), size(x,N+2))
  xsize = Cint[size(x,i) for i=1:N+1]
  xsize[N+1] *= size(x, N+2)
  params = Cint[winsize..., f.stride..., f.pad...]
  ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint}), x, y, xsize, params)
  y
end

function outsize{N}(f::Window{N}, w::Array, x::Array)
  dims = Array(Int, N)
  for i = 1:N
    dims[i] = (size(x,i) + 2*f.pad[i] - size(w,i)) ÷ f.stride[i] + 1
  end
  dims
end

function ∇window!()
end

type Window2D
  w1::Int
  w2::Int
  s1::Int
  s2::Int
  p1::Int
  p2::Int

  function Window2D(w1, w2, s1, s2, p1=0, p2=0)
    (s1 > 0 && s2 > 0) || throw("stride must be > 0")
    new(w1, w2, s1, s2, p1, p2)
  end
end

fwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_FWD_F32_HANDLE
fwd_handle(f::Window2D, ::Type{Float64}) = WINDOW2D_FWD_F64_HANDLE
bwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_BWD_F32_HANDLE
bwd_handle(f::Window2D, ::Type{Float64}) = WINDOW2D_BWD_F64_HANDLE

function make_params(f::Window2D)
  Int32[f.w1, f.w2, f.s1, f.s2, f.p1, f.p2]
end

@compat function (f::Window2D)(args::Vector{Var})
  x = args[1]
  y, params = window2d(f, x.value)
  df(gy) = hasgrad(x) && ∇window2d!(f, params, x.value, x.grad, gy)
  Var(y, df, args)
end
@compat (f::Window2D)(x::Var) = forward(f, [x])

function window2d{T}(f::Window2D, x::Matrix{T})
  w1, w2, s1, s2, p1, p2 = f.w1, f.w2, f.s1, f.s2, f.p1, f.p2
  n1 = (size(x,1) + 2*p1 - w1) ÷ s1 + 1
  n2 = (size(x,2) + 2*p2 - w2) ÷ s2 + 1
  params = Int32[w1, w2, s1, s2, p1, p2]
  y = Array(T, w1*w2, n1*n2)
  ccall(fwd_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Cint, Cint),
    x, params, y, size(x,1), size(x,2))
  y, params
end

function ∇window2d!{T}(f::Window2D, params::Vector{Int32}, x::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  ccall(bwd_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(gx,1), size(gx,2))
end
