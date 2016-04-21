export Linear

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
x = rand(Float32,10,5)
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

@compat (f::Linear)(arg) = forward(f, arg)
function forward!(f::Linear, v::Variable)
  v.value = f.w.value * v[1].value .+ f.b.value
  v.backward! = () -> begin
    # dy / dx = w^T * gy, dy / dw = gy * x^T
    T = eltype(v.value)
    gy = v.grad
    w, gw, b, gb = f.w.value, f.w.grad, f.b.value, f.b.grad
    hasgrad(v[1]) && BLAS.gemm!('T', 'N', T(1), w, gy, T(1), v[1].grad)
    BLAS.gemm!('N', 'T', T(1), gy, v[1].value, T(1), gw)
    for offset = 1:length(b):length(gy)
      BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
    end
  end
end

function update!(opt::Optimizer, f::Linear)
  update!(opt, f.w.value, f.w.grad)
  update!(opt, f.b.value, f.b.grad)
end
