type Linear <: Functor
  w::Variable
  b::Variable
end

function Linear{T}(::Type{T}, size1::Int, size2::Int)
  r = randn(size1, size2) * sqrt(1 / size2)
  w = convert(Matrix{T}, r)
  b = fill(T(0.01), size1)
  w = Variable(w, zeros(w))
  b = Variable(b, zeros(b))
  Linear(w, b)
end

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1

function forward!(f::Linear, v::Variable)
  v.value = linear(f.w.value, f.b.value, v[1].value)
end

function linear{T}(w::Matrix{T}, b::Vector{T}, x::Matrix{T})
  y = Array(T, size(w,1), size(x,2))
  gemm!('N', 'N', T(1), w, x, T(0), y)
  broadcast!(+, y, b, y)
  y
end

function backward!(f::Linear, v::Variable)
  w, b, x = f.w, f.b, v[1]
  gx = ∇linear(w.value, b.value, v.grad, w.grad, b.grad)
  addgrad!(x, gx)
end

"""
d_y / d_x = w^T * gy
d_y / d_w = gy * x^T
d_y / d_b = 1
"""
function ∇linear{T}(w::Matrix{T}, b::Vector{T}, gy::Matrix{T}, gw::Matrix{T}, gb::Vector{T})
  gx = similar(x)
  gemm!('T', 'N', T(1), w, gy, T(0), gx)
  gemm!('N', 'T', T(1), gy, x, T(1), gw)
  sum!(gb, gy)
  gx
end

function update!(opt::Optimizer, f::Linear)
  update!(opt, f.w.value, f.w.grad)
  update!(opt, f.b.value, f.b.grad)
end
