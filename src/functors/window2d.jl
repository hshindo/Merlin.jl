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

function forward!(f::Window2D, v::Variable)
  y, params = window2d(f, v[1].value)
  v.value = y
  v.state = params
end

function window2d{T,N}(f::Window2D, x::AFArray{T,N})
  w, s, p = [f.winsize...], f.stride, f.padsize
  w[1] == -1 && (w[1] = size(x,1))
  w[2] == -1 && (w[2] = size(x,2))
  n1 = (size(x,1) + 2*p[1] - w[1]) ÷ s[1] + 1
  n2 = (size(x,2) + 2*p[2] - w[2]) ÷ s[2] + 1
  params = Int32[w..., s..., p...]
  # y = af_array[0]
  y = Ptr{Void}[0]
  ccall(fwd_handle(f,T), Void,
    (Ptr{Void}, Ptr{Cint}, Ptr{Void}, Cint, Cint),
    x, params, y, size(x,1), size(x,2))
  AFArray{T,N}(y[1]), params
end

function backward!(f::Window2D, v::Variable)
  gx = ∇window2d(f, v.state, v[1].value, v.grad)
  addgrad!(v[1], gx)
end

function ∇window2d{T,N}(f::Window2D, params::Vector{Int32}, x::AFArray{T,N}, gy::AFArray{T,N})
  # y = af_array[0]
  gx = Ptr{Void}[0]
  ccall(bwd_handle(f,T), Void,
    (Ptr{Cint}, Ptr{Void}, Ptr{Void}, Cint, Cint),
    params, gy, gx, size(x,1), size(x,2))
  AFArray{T,N}(gx[1])
end
