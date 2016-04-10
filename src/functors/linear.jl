export Linear
using Base.LinAlg.BLAS

"""
## Linear
Computes linear transformation a.k.a. affine transformation.

\$ f(x) = W^{\mathrm{T}}x + b \$

where \$W\$ is a weight matrix, \$b\$ is a bias vector.

<img src="../assets/linear.png" width="300px">

### Functions
- `Linear(w, b)`
- `Linear{T}(::Type{T}, insize::Int, outsize::Int)`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Linear(Float32, 10, 3)
y = f(x)
```
"""
type Linear <: Functor
  w::Variable
  b::Variable
end

function Linear{T}(::Type{T}, insize::Int, outsize::Int)
  #r = randn(outsize, insize) * sqrt(1 / insize)
  #r = rand(outsize, insize) * 0.001
  x = sqrt(6 / (outsize+insize))
  r = rand(outsize, insize) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outsize, 1)
  Linear(Variable(w,zeros(w)), Variable(b,zeros(b)))
end

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1

function forward(f::Linear, x)
  y = f.w.value * x .+ f.b.value
  backward! = (gxs, gy) -> ∇linear!(f, x, gxs[1], gy)
  #backward! = v -> ∇linear!(f, x, v[1].grad, v.grad)
  y, backward!
end

"""
dy / dx = w^T * gy
dy / dw = gy * x^T
dy / db = 1
"""
function ∇linear!{T}(f::Linear, x::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  w, gw, b, gb = f.w.value, f.w.grad, f.b.value, f.b.grad
  gemm!('T', 'N', T(1), w, gy, T(0), gx)
  gemm!('N', 'T', T(1), gy, x, T(1), gw)
  for offset = 1:length(b):length(gy)
    axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
end

#=
function ∇linear!{T}(w::Matrix{T}, b::Matrix{T}, x::Matrix{T}, gx::Matrix{T}, gy::Matrix{T}, gw::Matrix{T}, gb::Matrix{T})
  gemm!('T', 'N', T(1), w, gy, T(1), gx)
  gemm!('N', 'T', T(1), gy, x, T(1), gw)
  for offset = 1:length(b):length(gy)
    axpy!(length(b), T(1.0), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
end
=#

function update!(opt::Optimizer, f::Linear)
  update!(opt, f.w.value, f.w.grad)
  update!(opt, f.b.value, f.b.grad)
end
