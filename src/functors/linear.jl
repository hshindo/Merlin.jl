export linear, Linear

type Linear
  w::Var
  b::Var
end

function Linear{T}(::Type{T}, indim::Int, outdim::Int)
  x = sqrt(6 / (indim + outdim))
  r = rand(outdim, indim) * 2x - x
  w = convert(Matrix{T}, r)
  b = fill(T(0), outdim, 1)
  Linear(Var(w), Var(b))
end

@compat (f::Linear)(x::Var) = linear(f.w, x, f.b)

"""
    linear(w::Var, x::Var, [b::Var])

Compute linear function (a.k.a. affine transformation).

```math
f(x) = w^{\mathrm{T}}x + b
```

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Linear(Float32,10,7)
y = f(x)
```
"""
function linear(w::Var, x::Var, b::Var)
  y = w.value * x.value
  broadcast!(.+, y, b.value)
  f(gy) = ∇linear!(w, x, b, gy)
  Var(y, nothing, f, [w,x,b])
end

function ∇linear!{T}(w::Var, x::Var, b::Var, gy::Array{T})
  hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x.grad, T(1), w.grad)
  hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.value, gy, T(1), x.grad)
  #s = sum(gy, 2)
  #broadcast!(+, gb, s)
end

#=
function ∇linear!(w::Var, b::Var, x::Var)
  w, gw, b, gb = w.value, w.grad, b.value, b.grad
  hasgrad(v[1]) && BLAS.gemm!('T', 'N', T(1), w, gy, T(1), v[1].grad)
  BLAS.gemm!('N', 'T', T(1), gy, v[1].value, T(1), gw)
  for offset = 1:length(b):length(gy)
    BLAS.axpy!(length(b), T(1), pointer(gy,offset), stride(gy,1), pointer(gb), stride(gb,1))
  end
end
=#

mat(a::Array) = reshape(a, size(a, 1), length(a)÷size(a,1))
isvec(a::Array) = ndims(a) == 2 && size(a, 2) == 1
