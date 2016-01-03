type Linear <: Functor
  w
  b
  x
  y
end

Linear(w, b) = Linear(w, b, nothing, nothing)

function Linear{T}(::Type{T}, xlength::Int, ylength::Int)
  r = randn(ylength, xlength) * sqrt(1 / xlength)
  w = convert(Array{T}, r) |> Var
  b = fill(T(0.01), ylength) |> Var
  Linear(w, b)
end

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1

function forward!(f::Linear, x)
  f.x = x
  f.y == nothing && (f.y = default(x))
  y = resize!(f.y, size(f.w,1), size(x,2))
  linear!(f.w.value, f.b.value, x.value, y.value)
end

function linear!{T}(w::Matrix{T}, b::Vector{T}, x::Matrix{T}, y::Matrix{T})
  gemm!('N', 'N', T(1), w, x, T(0), y)
  broadcast!(+, y, b, y)
end

backward!(f::Linear) = ∇linear!(f.w, f.b, f.x, f.y)

"""
d_y / d_x = w^T * gy
d_y / d_w = gy * x^T
d_y / d_b = 1
"""
function ∇linear!{T}(varw::Var{T,2}, varb::Var{T,1}, varx::Var{T,2}, vary::Var{T,2})
  w, gw = data(varw)
  b, gb = data(varb)
  x, gx = data(varx)
  y, gy = data(vary)
  varx.fixed || gemm!('T', 'N', T(1), w, gy, T(1), gx)
  varw.fixed || gemm!('N', 'T', T(1), gy, x, T(1), gw)
  vary.fixed || sum!(gb, gy)
end

function optimize!(opt::Optimizer, l::Linear)
  update!(opt, l.weight, l.gradweight)
  update!(opt, l.bias, l.gradbias)
end
