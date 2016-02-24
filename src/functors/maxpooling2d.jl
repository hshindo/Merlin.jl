const MAXPOOLING2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpooling2d_fwd_f32)
const MAXPOOLING2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :maxpooling2d_bwd_f32)

type MaxPooling2D <: Functor
  w1::Int
  w2::Int
  s1::Int
  s2::Int

  function MaxPooling2D(w1, w2, s1, s2)
    (s1 > 0 && s2 > 0) || error("stride must be > 0")
    new(w1, w2, s1, s2)
  end
end

fwd_handle(f::MaxPooling2D, ::Type{Float32}) = MAXPOOLING2D_FWD_F32_HANDLE
bwd_handle(f::MaxPooling2D, ::Type{Float32}) = MAXPOOLING2D_BWD_F32_HANDLE

function forward!(f::MaxPooling2D, v::Variable)
  y, maxind = maxpooling2d(f, v[1].value)
  v.value = y
  v.work = maxind
end

function maxpooling2d{T}(f::MaxPooling2D, x::Matrix{T})
  w1, w2, s1, s2 = f.w1, f.w2, f.s1, f.s2
  w1 < 0 && (w1 = size(x,1))
  w2 < 0 && (w2 = size(x,2))
  n1 = (size(x,1) - w1) ÷ s1 + 1
  n2 = (size(x,2) - w2) ÷ s2 + 1
  params = Int32[w1, w2, s1, s2]
  maxind = fill(Int32(-1), n1, n2)
  y = Array(T, n1, n2)
  ccall(fwd_handle(f,T), Void,
    (Ptr{T}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Cint, Cint),
    x, params, y, maxind, size(x,1), size(x,2))
  y, maxind
end

function backward!(f::MaxPooling2D, v::Variable)
  gx = ∇maxpooling2d(f, v[1].value, v.work, v.grad)
  addgrad!(v[1], gx)
end

function ∇maxpooling2d{T}(f::MaxPooling2D, x::Matrix{T}, maxind::Matrix{Int32}, gy::Matrix{T})
  gx = zeros(T, size(x))
  ccall(bwd_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint), maxind, gy, gx, length(gy))
  gx
end
