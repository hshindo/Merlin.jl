type MaxPool2D <: Functor
  w::Window2D

  function MaxPool2D(w1, w2, s1, s2, p1=0, p2=0)
    new(Window2D(w1, w2, s1, s2, p1, p2))
    #(s1 > 0 && s2 > 0) || throw("stride must be > 0")
    #new(w1, w2, s1, s2, p1, p2)
  end
end

function forward!(f::MaxPool2D, v::Variable)
  x = v[1].value

  #findmax(x, )

  y, maxind = maxpool2d(f, v[1].value)
  v.value = y
  v.state = maxind
end

function maxpool2d{T}(f::MaxPool2D, x::Matrix{T})
  w, s, p = [f.winsize...], f.stride, f.padsize
  w[1] == -1 && (w[1] = size(x,1))
  w[2] == -1 && (w[2] = size(x,2))
  n1 = (size(x,1) + 2*p[1] - w[1]) ÷ s[1] + 1
  n2 = (size(x,2) + 2*p[2] - w[2]) ÷ s[2] + 1
  params = Int32[w..., s..., p...]
  maxind = fill(Int32(-1), n1, n2)
  y = alloc_cpu(T, n1, n2)
  ccall(fwd_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Cint, Cint),
    x, params, y, maxind, size(x,1), size(x,2))
  y, maxind
end

const MAXPOOL2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpool2d_fwd_f32)
const MAXPOOL2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpool2d_bwd_f32)

type MaxPool2D <: Functor
  winsize::NTuple{2,Int}
  stride::NTuple{2,Int}
  padsize::NTuple{2,Int}

  function MaxPool2D(winsize, stride, padsize=(0,0))
    (stride[1] > 0 && stride[2] > 0) || error("stride must be > 0")
    new(winsize, stride, padsize)
  end
end

fwd_handle(f::MaxPool2D, ::Type{Float32}) = MAXPOOL2D_FWD_F32_HANDLE
bwd_handle(f::MaxPool2D, ::Type{Float32}) = MAXPOOL2D_BWD_F32_HANDLE

function forward!(f::MaxPool2D, v::Variable)
  y, maxind = maxpool2d(f, v[1].value)
  v.value = y
  v.state = maxind
end

function maxpool2d{T}(f::MaxPool2D, x::Matrix{T})
  w, s, p = [f.winsize...], f.stride, f.padsize
  w[1] == -1 && (w[1] = size(x,1))
  w[2] == -1 && (w[2] = size(x,2))
  n1 = (size(x,1) + 2*p[1] - w[1]) ÷ s[1] + 1
  n2 = (size(x,2) + 2*p[2] - w[2]) ÷ s[2] + 1
  params = Int32[w..., s..., p...]
  maxind = fill(Int32(-1), n1, n2)
  y = alloc_cpu(T, n1, n2)
  ccall(fwd_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Cint, Cint),
    x, params, y, maxind, size(x,1), size(x,2))
  y, maxind
end

function backward!(f::MaxPool2D, v::Variable)
  gx = ∇maxpool2d(f, v[1].value, v.state, v.grad)
  addgrad!(v[1], gx)
end

function ∇maxpool2d{T}(f::MaxPool2D, x::Matrix{T}, maxind::Matrix{Int32}, gy::Matrix{T})
  gx = zeros(T, size(x))
  ccall(bwd_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint), maxind, gy, gx, length(gy))
  gx
end
