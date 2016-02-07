type MaxPool2D <: Functor
  dim::Int
end

function forward!(f::MaxPool2D, v::Variable)
  x = v[1].value
  y, ind = findmax(x, f.dim)
  finalize(ind)
  v.value = y
  #v.state = ind
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
