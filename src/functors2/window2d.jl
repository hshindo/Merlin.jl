const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)

type Window2D <: Functor
  winsize::NTuple{2,Int}
  stride::NTuple{2,Int}
  padsize::NTuple{2,Int}
  params
  x
  y

  function Window2D(winsize, stride, padsize=(0,0))
    (stride[1] > 0 && stride[2] > 0) || error("stride must be > 0")
    new(winsize, stride, padsize, nothing, nothing, nothing)
  end
end

window2d_fwd_handle(::Type{Float32}) = WINDOW2D_FWD_F32_HANDLE
window2d_bwd_handle(::Type{Float32}) = WINDOW2D_BWD_F32_HANDLE

function forward!(f::Window2D)
  w, s, p = [f.winsize...], f.stride, f.padsize
  w[1] == -1 && (w[1] = size(x,1))
  w[2] == -1 && (w[2] = size(x,2))
  n1 = (size(x,1) + 2*p[1] - w[1]) ÷ s[1] + 1
  n2 = (size(x,2) + 2*p[2] - w[2]) ÷ s[2] + 1
  f.params = Int32[w..., s..., p...]
  y = resize!(f.y, prod(w), n1*n2)
  window2d!(f.params, f.x.value, y.value)
end

function window2d!{T}(params, x::Matrix{T}, y::Matrix{T})
  ccall(window2d_fwd_handle(T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Cint, Cint),
    x, params, y, size(x,1), size(x,2))
end

backward!(f::Window2D) = f.x.fixed || ∇window2d!(f.params, f.x.grad.value, f.y.grad.value)

function ∇window2d!{T}(params, gx::Matrix{T}, gy::Matrix{T})
  ccall(window2d_bwd_handle(T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(gx,1), size(gx,2))
end
