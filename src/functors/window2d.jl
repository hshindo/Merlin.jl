const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)

type Window2D <: Functor
  winsize::Tuple{Int,Int}
  stride::Tuple{Int,Int}
  padsize::Tuple{Int,Int}

  function Window2D(winsize, stride, padsize=(0,0))
    (stride[1] > 0 && stride[2] > 0) || error("stride must be > 0")
    new(winsize, stride, padsize)
  end
end

fwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_FWD_F32_HANDLE
bwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_BWD_F32_HANDLE

function forward{T}(f::Window2D, x::Matrix{T})
  w, s, p = [f.winsize...], f.stride, f.padsize
  w[1] == -1 && (w[1] = size(x,1))
  w[2] == -1 && (w[2] = size(x,2))
  n1 = (size(x,1) + 2*p[1] - w[1]) ÷ s[1] + 1
  n2 = (size(x,2) + 2*p[2] - w[2]) ÷ s[2] + 1
  params = Int32[w..., s..., p...]

  y = Array(T, prod(w), n1*n2)
  ccall(fwd_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Cint, Cint),
    x, params, y, size(x,1), size(x,2))
  y, (gy, gx) -> gx == nothing || backward!(f, params, gy, gx)
end

function backward!{T}(f::Window2D, params::Vector{Int32}, gy::Matrix{T}, gx::Matrix{T})
  ccall(bwd_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(gx, 1), size(gx, 2))
end
