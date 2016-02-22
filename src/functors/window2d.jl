const WINDOW2D_FWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_fwd_f32)
const WINDOW2D_BWD_F32_HANDLE = Libdl.dlsym(Native.library, :window2d_bwd_f32)

type Window2D <: Functor
  w1::Int
  w2::Int
  s1::Int
  s2::Int
  p1::Int
  p2::Int
  iscolumn::Bool

  function Window2D(w1, w2, s1, s2, p1=0, p2=0, iscolumn=true)
    #(s1 > 0 && s2 > 0) || throw("stride must be > 0")
    new(w1, w2, s1, s2, p1, p2, iscolumn)
  end
end

fwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_FWD_F32_HANDLE
bwd_handle(f::Window2D, ::Type{Float32}) = WINDOW2D_BWD_F32_HANDLE

function getparam(f::Window2D, x)
  xsize = size(x)
  w1, w2 = f.w1, f.w2
  w1 == -1 && (w1 = xsize[1])
  w2 == -1 && (w2 = (ndims(x) == 1) ? 1 : xsize[2])
  w1, w2
end

function forward!(f::Window2D, v::Variable)
  x = v[1].value
  w1, w2 = getparam(f, x)
  y = unwrap(x, w1, w2, f.s1, f.s2, f.p1, f.p2, f.iscolumn)
  v.value = y
end

function backward!(f::Window2D, v::Variable)
  x = v[1].value
  xsize = size(x)
  w1, w2 = getparam(f, x)

  gx = wrap(v.grad, xsize[1], xsize[2], w1, w2, f.s1, f.s2, f.p1, f.p2, f.iscolumn)
  addgrad!(v[1], gx)
  #addgrad!(v[1], zeros(v[1].value))
end

function getraw(in::AFArray)
  p = device_ptr(in)
  pp = convert(Ptr{Float32}, p)
  pointer_to_array(pp, jl_size(in))
end

function backward2!(f::Window2D, v::Variable)
  x = getraw(v[1].value)
  #gy = getraw(v.grad)
  #ndims(x) == 1 && (x = reshape(x, (length(x),1)))
  #ndims(gy) == 1 && (gy = reshape(gy, (length(gy),1)))
  #gx = ∇window2d(f, x, gy)
  v[1].grad = zeros(v[1].value)
  #addgrad!(v[1], gx)
end

function ∇window2d{T}(f::Window2D, x::Matrix{T}, gy::Matrix{T})
  w1, w2, s1, s2, p1, p2 = f.w1, f.w2, f.s1, f.s2, f.p1, f.p2
  w1 == -1 && (w1 = size(x,1))
  w2 == -1 && (w2 = size(x,2))
  n1 = (size(x,1) + 2*p1 - w1) ÷ s1 + 1
  n2 = (size(x,2) + 2*p2 - w2) ÷ s2 + 1
  params = Int32[w1, w2, s1, s2, p1, p2]

  gx = zeros(T, size(x))
  #ccall(bwd_handle(f,T), Void,
  #  (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
  #  params, gy, gx, size(x,1), size(x,2))
  gx
end
