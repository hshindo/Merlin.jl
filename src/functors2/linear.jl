export Linear

"""
## Linear

Compute linear transformation a.k.a. affine transformation.

```math
f(x) = W^{T}x + b
```

where \$W\$ is a weight matrix, \$b\$ is a bias vector.

<div align="center"><img src="../assets/linear.png" width="300px"></div>

### Arguments
* `Linear(w,b)`
* `Linear{T}(::Type{T}, insize::Int, outsize::Int)`

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
  x = sqrt(6 / (outsize+insize))
  r = rand(outsize, insize) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outsize, 1)
  Linear(Variable(w,zeros(w)), Variable(b,zeros(b)))
end

#mat(a::Array) = reshape(a, size(a,1), length(a)÷size(a,1))
#isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1

@compat function (f::Linear){T}(x::Variable{T})
  y = f.w.val * x.val .+ f.b.val
  backward! = y -> backward!(f, x, y)
  Variable(y, f, (x,), backward!)
end

function backward!{T}(f::Linear, x::Variable{Matrix{T}}, y::Variable{Matrix{T}})
  w, b = f.w, f.b
  gy = get(y.grad)
  isnull(x.grad) || BLAS.gemm!('T', 'N', T(1), w.val, gy, T(1), x.grad)
  isnull(w.grad) || BLAS.gemm!('N', 'T', T(1), gy, x.val, T(1), w.grad)
  if !isnull(b.grad)
    gb = get(b.grad)
    for offset = 1:length(gb):length(gy)
      BLAS.axpy!(length(gb), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
    end
  end
end
