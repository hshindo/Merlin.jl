type Linear <: Functor
  w::Variable
  b::Variable
end

function Linear{T}(::Type{T}, size1::Int, size2::Int)
  w = randn(size2, size1) * sqrt(1 / size1)
  w = convert(Matrix{T}, w) |> AFArray
  b = fill(AFArray, T(0.01), size2)
  Linear(Variable(w), Variable(b))
end

function forward!(f::Linear, v::Variable)
  x = v[1].value
  w, b = f.w.value, f.b.value
  v.value = w * x + b
end

function backward!(f::Linear, v::Variable)
  x = v[1]
  w, b = f.w, f.b
  gy = v.grad
  addgrad!(w, A_mul_Bt(gy, x.value))
  addgrad!(b, sum(gy,2))
  addgrad!(x, At_mul_B(w.value, gy))
end

function linear{T}(w::Matrix{T}, b::Vector{T}, x::Matrix{T})
  y = alloc_cpu(T, size(w,1), size(x,2))
  gemm!('N', 'N', T(1), w, x, T(0), y)
  broadcast!(+, y, b, y)
  y
end

function backward2!(f::Linear, v::Variable)
  ∇linear_dwb!(f.w.grad, f.b.grad, v[1].value, v.grad)
  gx = ∇linear_dx(f.w.value, v[1].value, v.grad)
  addgrad!(v[1], gx)
end

"""
d_y / d_x = w^T * gy
d_y / d_w = gy * x^T
d_y / d_b = 1
"""
function ∇linear_dx{T}(w::Matrix{T}, x::Matrix{T}, gy::Matrix{T})
  gx = alloc_cpu(T, size(x))
  gemm!('T', 'N', T(1), w, gy, T(0), gx)
  gx
end

function ∇linear_dwb!{T}(gw::Matrix{T}, gb::Vector{T}, x::Matrix{T}, gy::Matrix{T})
  gemm!('N', 'T', T(1), gy, x, T(1), gw)
  sum!(gb, gy)
end

function optimize!(opt::Optimizer, f::Linear)
  jl_size(f.w.value) == (50,50) && return
  update!(opt, f.w)
  #update!(opt, f.b)
end
