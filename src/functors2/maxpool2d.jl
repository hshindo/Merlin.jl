const MAXPOOL2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpool2d_fwd_f32)
const MAXPOOL2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpool2d_bwd_f32)

type MaxPool2D <: Functor
  winsize::NTuple{2,Int}
  stride::NTuple{2,Int}
  padsize::NTuple{2,Int}
  params
  maxind
  x
  y

  function MaxPool2D(winsize, stride, padsize=(0,0))
    (stride[1] > 0 && stride[2] > 0) || error("stride must be > 0")
    new(winsize, stride, padsize, nothing, nothing, nothing, nothing)
  end
end

maxpool2d_fwd_handle(::Type{Float32}) = MAXPOOL2D_FWD_F32_HANDLE
maxpool2d_bwd_handle(::Type{Float32}) = MAXPOOL2D_BWD_F32_HANDLE

function forward!(f::MaxPool2D)
  w, s, p = [f.winsize...], f.stride, f.padsize
  w[1] == -1 && (w[1] = size(x,1))
  w[2] == -1 && (w[2] = size(x,2))
  n1 = (size(x,1) + 2*p[1] - w[1]) ÷ s[1] + 1
  n2 = (size(x,2) + 2*p[2] - w[2]) ÷ s[2] + 1
  f.params = Int32[w..., s..., p...]
  f.maxind = fill(Int32(-1), n1, n2)
  y = resize!(f.y, n1, n2)
  maxpool2d!(f.params, f.maxind, x.value, y.value)
end

function maxpool2d!{T}(params, maxind, x::Matrix{T}, y::Matrix{T})
  ccall(maxpool2d_fwd_handle(T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Cint, Cint),
    x, params, y, maxind, size(x,1), size(x,2))
  y
end

backward!(f::MaxPool2D) = f.x.fixed || ∇maxpool2d!(f.params, f.maxind, f.x.grad.value, f.y.grad.value)

function ∇maxpool2d!{T}(f::MaxPool2D, maxind, gx::Matrix{T}, gy::Matrix{T})
  ccall(maxpool2d_bwd_handle(T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint), maxind, gy, gx, length(gy))
end
