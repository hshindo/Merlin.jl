export Linear
using Base.LinAlg.BLAS

"""
## Linear
Computes linear transformation a.k.a. affine transformation.

\\[
  f(x) = wx
\\]

\$f(x) = W^{\mathrm{T}}x + b\$

where \(W\) is a weight matrix, \(b\) is a bias vector.

### Functions
- `Linear(w, b)`
- `Linear{T}(::Type{T}, insize::Int, outsize::Int)`

### 👉 Example
```julia
#x = Variable(rand(Float32,10,5))
#f = Linear(Float32, 10, 3)
#y = f(x)
```
"""
type Linear <: Functor
  w::Variable
  b::Variable
end

function Linear{T}(::Type{T}, insize::Int, outsize::Int)
  #r = randn(outsize, insize) * sqrt(1 / insize)
  #r = rand(outsize, insize) * 0.001
  b = sqrt(6 / (outsize+insize))
  r = rand(outsize, insize) * 2b - b
  w = convert(Matrix{T}, r)
  b = fill(T(0), outsize, 1)
  w = Variable(w, zeros(w))
  b = Variable(b, zeros(b))
  Linear(w, b)
end

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1

#function call(f::Linear, v::Variable)
#  f.w * v + f.b
#end

function forward!(f::Linear, v::Variable)
  v.value = linear(f.w.value, f.b.value, v[1].value)
  v.backward! = () -> begin
    v[1].grad == nothing && (v[1].grad = zeros(v[1].value))
    ∇linear!(f.w.value, f.b.value, v[1].value, v[1].grad, v.grad, f.w.grad, f.b.grad)
  end
end

function linear{T}(w::Matrix{T}, b::Matrix{T}, x::Matrix{T})
  y = w * x
  broadcast!(+, y, b, y)
  y
end

"""
dy / dx = w^T * gy
dy / dw = gy * x^T
dy / db = 1
"""
function ∇linear!{T}(w::Matrix{T}, b::Matrix{T}, x::Matrix{T}, gx::Matrix{T}, gy::Matrix{T}, gw::Matrix{T}, gb::Matrix{T})
  gemm!('T', 'N', T(1), w, gy, T(1), gx)
  gemm!('N', 'T', T(1), gy, x, T(1), gw)
  #axpy!(T(1), reducedim(+, gy, 2), gb)
  for offset = 1:length(b):length(gy)
    axpy!(length(b), T(1.0), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
  #for j = 1:size(gy,2)
  #  @inbounds @simd for i = 1:size(gy,1)
  #    gb[i] += gy[i,j]
  #  end
  #end
end

function update!(opt::Optimizer, f::Linear)
  update!(opt, f.w.value, f.w.grad)
  update!(opt, f.b.value, f.b.grad)
end
