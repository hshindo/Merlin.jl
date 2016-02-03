type Window2D <: Functor
  w1::Int
  w2::Int
  s1::Int
  s2::Int
  p1::Int
  p2::Int

  function Window2D(w1, w2, s1, s2, p1=0, p2=0)
    #(s1 > 0 && s2 > 0) || throw("stride must be > 0")
    new(w1, w2, s1, s2, p1, p2)
  end
end

function forward!(f::Window2D, v::Variable)
  w1, w2 = f.w1, f.w2
  x = v[1].value
  println("xsize: $(size(x))")
  w1 == -1 && (w1 = size(x,1))
  w2 == -1 && (w2 = size(x,2))
  println("w1, w2: $(w1), $(w2)")
  x = v[1].value
  y = unwrap(x, w1, w2, f.s1, f.s2, f.p1, f.p2)
  v.value = y
  println("window done")
end

function backward!(f::Window2D, v::Variable)
  gx = ∇window2d(f, v.state, v[1].value, v.grad)
  addgrad!(v[1], gx)
end

function ∇window2d{T}(f::Window2D, params::Vector{Int32}, x::Matrix{T}, gy::Matrix{T})
  gx = zeros(T, size(x))
  ccall(bwd_handle(f,T), Void,
    (Ptr{Cint}, Ptr{T}, Ptr{T}, Cint, Cint),
    params, gy, gx, size(x,1), size(x,2))
  gx
end
